import random

def dataProcess(src_filename):
    ffp = open('female_names.txt', 'r')
    mfp = open('male_names.txt', 'r')
    namesF = ffp.readlines()
    namesM = mfp.readlines()
    for i in range(len(namesF)):
        namesF[i] = namesF[i][:-1]
    for i in range(len(namesM)):
        namesM[i] = namesM[i][:-1]
    with open(src_filename, 'w') as fp:
        for name in namesM:
            fp.write(name + ', 1\n')
        for name in namesF:
            fp.write(name + ', 0\n')
    fp.close()

def divideData(params):
    
    Basic_Filename = params['src_filename']
    train_rate = params['train_rate']
    valid_rate = params['valid_rate']
    
    fp = open(Basic_Filename, 'r')
    temp = fp.readlines()
    dataScale = len(temp)
    
    random.shuffle(temp)
    train_index = int(dataScale * train_rate)
    valid_index = int(dataScale * valid_rate)
    
    train_data = temp[: train_index]
    valid_data = temp[train_index : valid_index]
    test_data = temp[valid_index :]
    
    trainfp = open(params['train_filename'], 'w')
    validfp = open(params['valid_filename'], 'w')
    testfp = open(params['test_filename'], 'w')
    for ele in train_data:
        trainfp.write(ele)
    for ele in valid_data:
        validfp.write(ele)
    for ele in test_data:
        testfp.write(ele)
    testfp.close()
    validfp.close()
    trainfp.close()
    fp.close()
    
def load_Local_Data(params):
    
    print('Preparing local data...')
    
    dataProcess(params['src_filename'])
    divideData(params)
    
    print('Local data ready.\n')