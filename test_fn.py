import torch
from torch.utils.data import DataLoader
from utils.generic import freeze_model
from copy import deepcopy
from utils.viz import plt_test_confmat, plt_confmats, plt_test_confmat_task
from parse import args
import pickle as pkl



def get_stat_exp(y,y_hats,exp_idx, task_id,task_predictions):
    """ Compute accuracy and task accuracy for each experience."""
    conf_mat = torch.zeros((exp_idx+1, exp_idx+1))

    for i in range(exp_idx+1):
        ybuff= y[task_id==i]
        y_hats_buff=y_hats[task_id==i]
        acc = (ybuff==y_hats_buff).sum()/y_hats_buff.shape[0]

        for j in range(exp_idx+1):
            conf_mat[i,j] = ((task_id==i)&(task_predictions==j)).sum()/(task_id==i).sum()

        print(f"EXP:{i}, acc:{acc:.3f}, task:{conf_mat[i,i]:.3f}, distrib:{[round(conf_mat[i,j].item(), 3) for j in range(exp_idx+1)]}")


def entropy(vec):
    return -torch.sum(vec * torch.log(vec + 1e-7), dim=1)



def test(strategy, test_set, plot=True):
    strategy.model.eval()
    dataloader = DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=8)

    confusion_mat = torch.zeros((args.n_classes,args.n_classes))
    confusion_mat_e = torch.zeros((args.n_classes,args.n_classes))
    confusion_mat_taw = torch.zeros((args.n_classes,args.n_classes))

    y_hats = []
    y_hats_e = []
    y_taw = []
    ys = []
    task_predictions = []
    task_ids = []
    for i, (x, y, task_id) in enumerate(dataloader):
        frag_preds = []
        entropy_frag = []
        for j in range(strategy.experience_idx+1):
            # create a temporary model copy
            model = freeze_model(deepcopy(strategy.model))

            strategy.pruner.set_gating_masks(model, j, weight_sharing=args.weight_sharing, distillation=True)
            model.load_bn_params(j)
            model.exp_idx = j

            pred = model(x.to(args.device))
            if not args.dataset == 'CORE50':
                pred = pred[:, j*args.classes_per_exp:(j+1)*args.classes_per_exp]

            frag_preds.append(torch.softmax(pred/args.temperature, dim=1))
            entropy_frag.append(entropy(torch.softmax(pred/args.temperature, dim=1)))

        #on_shell_confidence, elsewhere_confidence = confidence(frag_preds, task_id)
        #print(f"on_shell confidence:{[round(c.item(), 2) for c in on_shell_confidence]}\nelsewhere confidence:{[round(c.item(), 2) for c in elsewhere_confidence]}")

        frag_preds = torch.stack(frag_preds)  # [n_frag, bsize, n_classes]
        entropy_frag = torch.stack(entropy_frag)  # [bsize, n_frag]

        batch_size = frag_preds.shape[1]

        ### select across the top 2 of likelihood the head  with the lowest entropy
        # buff -> batch_size  x 2, 0-99 val 
        buff = frag_preds.max(dim=-1)[0].argsort(dim=0)[-2:] # [2, bsize]


        # buff_entropy ->  2 x batch_size, entropy values
        indices = torch.arange(batch_size)
        if buff.shape[0] == 1:
            buff_entropy = entropy_frag[buff[0, :], indices].unsqueeze(0)
        else:
            a = entropy_frag[buff[0, :], indices]
            b = entropy_frag[buff[1, :], indices]
            buff_entropy = torch.stack((a, b)) # [2, bsize]

        # index of min entropy ->  1 x batch_size, val 0-1
        index_min = buff_entropy.argsort(dim=0)[0,:]

        index_class = buff[index_min, indices]

        task_predictions.append(buff[-1])
        if args.dataset == 'CORE50':
            y_hats.append(frag_preds[buff[-1], indices].argmax(dim=1))
            y_hats_e.append(frag_preds[index_class, indices].argmax(dim=1))
            y_taw.append(frag_preds[task_id.to(torch.int32), indices].argmax(dim=-1))

        else:
            y_hats_e.append(frag_preds[index_class, indices].argmax(dim=1)+ args.classes_per_exp*index_class)
            y_hats.append(frag_preds[buff[-1], indices].argmax(dim=1) + args.classes_per_exp*buff[-1])
            y_taw.append(frag_preds[task_id.to(torch.int32), indices].argmax(dim=-1) + (args.classes_per_exp*task_id.to(args.cuda)).to(torch.int32))

        task_ids.append(task_id)
        ys.append(y)

    # concat labels and preds
    y_hats = torch.cat(y_hats, dim=0).to('cpu')
    y_hats_e = torch.cat(y_hats_e, dim=0).to('cpu')
    y = torch.cat(ys, dim=0).to('cpu')
    y_taw = torch.cat(y_taw, dim=0).to('cpu')
    task_predictions = torch.cat(task_predictions, dim=0).to('cpu')
    task_ids = torch.cat(task_ids, dim=0).to('cpu')


    # assign +1 to the confusion matrix for each prediction that matches the label
    for i in range(y.shape[0]):
        confusion_mat[y[i], y_hats[i]] += 1
        confusion_mat_e[y[i], y_hats_e[i]] += 1
        confusion_mat_taw[y[i], y_taw[i]] += 1

    
    #task confusion matrix and forgetting mat
    for j in range(strategy.experience_idx+1):
        i = strategy.experience_idx
        acc_conf_mat_task = confusion_mat[j*args.classes_per_exp:(j+1)*args.classes_per_exp, j*args.classes_per_exp:(j+1)*args.classes_per_exp].diag().sum()/confusion_mat[i*args.classes_per_exp:(i+1)*args.classes_per_exp,:].sum()
        strategy.confusion_mat_task[i][j] = acc_conf_mat_task
        strategy.forgetting_mat[i][j] = strategy.confusion_mat_task[:, j].max()-acc_conf_mat_task


    
        
    # compute accuracy
    accuracy = confusion_mat.diag().sum() / confusion_mat.sum()
    accuracy_e = confusion_mat_e.diag().sum() / confusion_mat_e.sum()
    accuracy_taw = confusion_mat_taw.diag().sum() / confusion_mat_taw.sum()


    task_accuracy = (task_predictions==task_ids).sum()/y_hats.shape[0]
    print(f"Test Accuracy: {accuracy:.4f},Test Accuracy with entropy: {accuracy_e:.4f},Test Accuracy taw: {accuracy_taw:.4f}, Task accuracy: {task_accuracy:.4f}")
    get_stat_exp(y, y_hats, strategy.experience_idx, task_ids,task_predictions)

    if plot:
        plt_test_confmat(args.run_name, confusion_mat, strategy.experience_idx)
        if strategy.experience_idx == args.n_experiences-1:
            plt_test_confmat_task(args.run_name, strategy.confusion_mat_task)
            torch.save(strategy.forgetting_mat, f'./logs/{args.run_name}/forgetting_mat.pt')
            torch.save(strategy.confusion_mat_task, f'./logs/{args.run_name}/confusion_mat_task.pt')


    if strategy.experience_idx == args.n_experiences-1:
        res = {}
        res['y'] = y.cpu().numpy()
        res['y_hats'] = y_hats.cpu().numpy()
        res['y_hats_e'] = y_hats_e.cpu().numpy()
        res['frag_preds'] = frag_preds.cpu().numpy()
        res['entropy_frag'] = entropy_frag.cpu().numpy()
        res['y_taw'] = y_taw.cpu().numpy()

        # write to file
        with open(f'./logs/{args.run_name}/res.pkl', 'wb') as f:
            pkl.dump(res, f)

    return accuracy, task_accuracy, accuracy_e, accuracy_taw


#################### MIND TESTS ####################

def test_single_exp(pruner, tested_model, loader, exp_idx, distillation):
    confusion_mat = torch.zeros((args.n_classes, args.n_classes))
    y_hats = []
    ys = []
    for i, (x, y, _) in enumerate(loader):
        model = freeze_model(deepcopy(tested_model))
        preds = torch.softmax(model(x.to(args.device)), dim=1)

        #frag preds size = (hid,bsize,100) and I want to reshape (bsize, hid, 100)
        y_hats.append(preds.argmax(dim=1))
        ys.append(y)

    # concat labels and preds
    y_hats = torch.cat(y_hats, dim=0).to('cpu')
    y = torch.cat(ys, dim=0).to('cpu')

    # assign +1 to the confusion matrix for each prediction that matches the label
    for i in range(y.shape[0]):
        confusion_mat[y[i], y_hats[i]] += 1

    return confusion_mat


def test_during_training(pruner, train_dloader, test_dloader, model, fresh_model, scheduler, epoch, exp_idx, distillation, plot=True):
    if distillation:
        model = model.eval()
    else:
        model = fresh_model.eval()

    with torch.no_grad():
        train_conf_mat = test_single_exp(pruner, model, train_dloader, exp_idx, distillation)
        test_conf_mat = test_single_exp(pruner, model, test_dloader, exp_idx, distillation)
        # compute accuracy
        train_acc = train_conf_mat.diag().sum() / train_conf_mat.sum()
        test_acc = test_conf_mat.diag().sum() / test_conf_mat.sum()
        print(f"    e:{epoch:03}, tr_acc:{train_acc:.4f}, te_acc:{test_acc:.4f} lr:{scheduler.get_last_lr()[0]:.5f}")

        if plot:
            plt_confmats(args.run_name, train_conf_mat, test_conf_mat, distillation, exp_idx)

    model.train()

    return train_acc, test_acc


def confidence(frag_preds, task_id):
    on_shell_probs = []
    elsewhere_probs = []
    for i, frag in enumerate(frag_preds):
        on_shell_probs.append(torch.softmax(frag[task_id==i], dim = -1))
        elsewhere_probs.append(torch.softmax(frag[task_id!=i], dim = -1))


    max_on_shell_probs = torch.max(torch.stack(on_shell_probs), dim = -1)[0]
    on_shell_confidence = [(1./(1.-p + 1e-6).mean()) for p in max_on_shell_probs]

    max_elsewhere_probs = torch.max(torch.stack(elsewhere_probs), dim = -1)[0]
    elsewhere_confidence = [(1./(1.-p + 1e-6).mean()) for p in max_elsewhere_probs]

    return on_shell_confidence, elsewhere_confidence
