import torchvision.transforms as transforms


default_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        ),
    ]


custom_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomChoice([
        transforms.ColorJitter(brightness=.5,hue=.3),
        transforms.GaussianBlur(kernel_size=5,sigma=(1,2.5)),
        transforms.RandomRotation(degrees=(0,180))
        ]),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        ),
    ]

to_tensor_and_normalize = [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        ),
    ]


default_transforms_core50 = [
        #transforms.RandomCrop(128, padding=16),
        transforms.RandomCrop(64, padding=4),
        #transforms.RandomGrayscale(p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.5,hue=.3),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5998523831367493, 0.5575963854789734, 0.5395311713218689), (0.20457075536251068, 0.2166813313961029, 0.22945666313171387)
        ),
    ]

to_tensor_and_normalize_core50 = [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5998523831367493, 0.5575963854789734, 0.5395311713218689), (0.20457075536251068, 0.2166813313961029, 0.22945666313171387)
        ),
    ]

default_transforms_TinyImageNet = [
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        ),
    ]

to_tensor_and_normalize_TinyImageNet = [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        )
    ]


default_transforms_Synbols = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.47957573372395784, 0.4786930207950382, 0.4795725401730997), (0.2840923785401597, 0.28447272496390646, 0.28412646131981306)
        )
        ]




to_tensor_and_normalize_Synbols = [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.47957573372395784, 0.4786930207950382, 0.4795725401730997), (0.2840923785401597, 0.28447272496390646, 0.28412646131981306)
        )
    ]
