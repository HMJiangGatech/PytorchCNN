ARCH = "orth_resnet20"
DATASET = "cifar10"
LEARNING_RATE = 0.1
MOMENTUM_RATE = 0.9
WEIGHT_DECAY = 5e-4
EPOCHES = 164
LRDECAY_EPOCHES = None
LRDECAY_SCHEDULE = [81, 122]
RANDOMSEED = 2018

# Show info
SHOW_PROGRESS = False
PRINT_FREQ = 78
SHOW_SV_INFO = False
SHOW_SV_EPOCH = []

# Orth Parameters
ORTH_REG = 0.1
WEIGHT_DECAY = 0

main()
DATASET = "cifar100"
main()
