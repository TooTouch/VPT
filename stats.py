dataset_stats = {
    "imagenet":{
        "num_classes" : 1000,
        "img_size"    : 224,
        "mean"        : (0.485, 0.456, 0.406),
        "std"         : (0.229, 0.224, 0.225)
    },
    "cifar10":{
        "num_classes" : 10,
        "img_size"    : 32,
        "mean"        : (0.4914, 0.4822, 0.4465),
        "std"         : (0.247, 0.2435, 0.2616)
    },
    "cifar100":{
        "num_classes" : 100,
        "img_size"    : 32,
        "mean"        : (0.5071, 0.4867, 0.4408),
        "std"         : (0.2675, 0.2565, 0.2761)
    },
    "svhn":{
        "num_classes" : 10,
        "img_size"    : 32,
        "mean"        : (0.4377, 0.4438, 0.4728), 
        "std"         : (0.1980, 0.2010, 0.1970)
    },
    "tiny_imagenet_200":{
        "num_classes" : 200,
        "img_size"    : 64,
        "mean"        : (0.4802, 0.4481, 0.3975), 
        "std"         : (0.2764, 0.2689, 0.2816)
    }
}