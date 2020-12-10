from dvn import FCN
import numpy as np
import dvn.misc as ms
import dvn.losses as ls
import dvn.utils as ut
import glob

if len(sys.argv) > 2:

    path = sys.argv[1]
    savepath = sys.argv[2]
    data = []
    file_list = glob.glob(path+'/imagesTr/*.gz')
    file_list.sort()
    print(file_list)
    i=0
    for f in file_list[0:4]:
        i=i+1
        img=ut.get_itk_image(f)
        ut.get_itk_data(img, True)
        data.append(ut.get_itk_array(img))
        #print(i)
      
    dataseg = []
    file_list = glob.glob(path+'/labelsTr/*.gz')
    n = len(file_list)
    file_list.sort()
    print(file_list)
    for f in file_list[0:4]:
        img=ut.get_itk_image(f)
        ut.get_itk_data(img, True)
        dataseg.append(ut.get_itk_array(img))
    
    dim = 3
    net = FCN(cross_hair=True, dim=dim)
    net.compile(loss=ls.weighted_categorical_crossentropy_with_fpr())
    #N = (10, 1,) +(64,)*dim
    X = np.array(data)
    a1, a2, a3, a4 = X.shape
    #X2 = np.array([1, 2])
    X2=np.reshape(X,(a1,1,a2, a3, a4))
    print(X2.shape)
    #Y = np.random.randint(2, size=N)
    Y = np.squeeze(np.array(dataseg))
    print(Y.shape)
    Y = ms.to_one_hot(Y)
    print(Y.shape)
    Y = np.transpose(Y, axes=[0,dim+1] + list(range(1,dim+1)))
    print(Y.shape)
    print('Testing FCN Network')
    print('Data Information => ', 'volume size:', X.shape, ' labels:',np.unique(Y))
    net.fit(x=X2, y=Y, epochs=30, batch_size=1, shuffle=True)
    net.save(filename=savepath+'/model.dat')
    #trainGen = csv_image_generator('/content/gdrive/My Drive/DVN/deepvesselnet/datasyn', bs=1, mode="train")
    #net.fit_generator(trainGen, **{'steps_per_epoch':n , 'epochs':30, 'batch_size':1, 'shuffle':True})
