from keras.models import load_model
import screenGrabber
from trainer import imageprocessing
import numpy as np
from keySender import PressKey
from keySender import ReleaseKey
from keySender import PressAndReleaseKey
import time
import cv2
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 150, 400, 1
model = load_model("model-010.h5")
W = 0x11
A = 0x1E
S = 0x1F
D = 0x20
UP = 0x57
LEFT = 0x61
DOWN = 0x62
RIGHT = 0x63
SPACE = 0x20

for j in range(5):
    print(j + 1)
    time.sleep(1)
k = np.array([[0,0,0]])
while True:
    image1 = screenGrabber.grab_screen([0, 350, 800, 650])
    image1 = imageprocessing(image1)
    image1 = cv2.resize(image1, (400, 150))
    #new_img = imageprocessing(new_img)
    new1 = image1
    image1 = np.array(image1, dtype=np.uint8)
    image1 = image1.reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    image1 = image1.astype('float32')
    image1 /= 255

    output = (model.predict(image1,batch_size=1))
    #output = np.abs(output)
    print(output)
    
    if output[0][0] > output[0][1] and output[0][0] > output[0][2] and output[0][0] > output[0][3]:
        op = output[0][0]
        os = output[0][3]
        if op > 0.8:
            PressKey(LEFT)
            ReleaseKey(DOWN)
            ReleaseKey(RIGHT)
            ReleaseKey()
            print("LEFT")
        elif os > 0.6:
            PressKey(LEFT)
            PressAndReleaseKey(SPACE, 1.5)
            ReleaseKey(DOWN)
            ReleaseKey(RIGHT)
            ReleaseKey()
            print("lNitro")
        else:
            print("AS")
            PressKey(LEFT)
            PressKey(DOWN)
            ReleaseKey(RIGHT)
    elif output[0][2] > output[0][1] and output[0][2] > output[0][3]:
        op = output[0][2]
        os = output[0][3]
        if op > 0.8:
            print("RIGHT")
            PressKey(RIGHT)
            ReleaseKey(DOWN)
            ReleaseKey(LEFT)
        elif os > 0.6:
            PressKey(RIGHT)
            PressAndReleaseKey(SPACE, 1.5)
            ReleaseKey(DOWN)
            ReleaseKey(LEFT)
            ReleaseKey()
            print("rNitro")
        else:
            print("DS")
            PressKey(RIGHT)
            PressKey(DOWN)
            ReleaseKey(LEFT)
    elif output[0][3] > output[0][1]:
        PressAndReleaseKey(SPACE, 1.5)
        ReleaseKey(RIGHT)
        ReleaseKey(DOWN)
        ReleaseKey(LEFT)
        ReleaseKey()
        print("Nitro")
    elif output[0][1] > output[0][3]:
        PressKey(DOWN)
        ReleaseKey(LEFT)
        ReleaseKey(RIGHT)
        print('DOWN')
    else:
        time.sleep(0.1)


    '''if output[0][0] > output[0][1] and output[0][0] > output[0][2]:
        op = output[0][0]
        if op > 0.8:
            PressKey(A)
            ReleaseKey(W)
            ReleaseKey(D)
            print("A")
        else:
            print("AW")
            PressKey(A)
            PressKey(W)
            ReleaseKey(D)
    elif output[0][2] > output[0][1]:
        op = output[0][2]
        if op > 0.8:
            print("D")
            PressKey(D)
            ReleaseKey(W)
            ReleaseKey(A)
        else:
            print("DW")
            PressKey(D)
            PressKey(W)
            ReleaseKey(A)
    else:
        PressKey(W)
        ReleaseKey(A)
        ReleaseKey(D)
        print('W')'''

    #time.sleep(0.2)


    '''out = (output-k)
    out[0][0] = (out[0][0]/k[0][0])*100000
    out[0][1] = (out[0][1] / k[0][1]) * 100000
    out[0][2] = (out[0][2] / k[0][2]) * 100000
    #print(out)
    if out[0][0] > out[0][1] and out[0][0] > out[0][2] or (out[0][0] < 50 and out[0][0] > 0):
        op = out[0][0]
        if op > 10:
            PressKey(A)
            ReleaseKey(W)
            ReleaseKey(D)
            print("AW")
        else:
            print("A")
            PressKey(A)
            PressKey(W)
            ReleaseKey(D)

    elif out[0][2] > out[0][1] or (out[0][2] < 50 and out[0][2] > 0):
        op = out[0][2]
        if op < -10:
            print("DW")
            PressKey(D)
            ReleaseKey(W)
            ReleaseKey(A)
        else:
            print("D")
            PressKey(D)
            PressKey(W)
            ReleaseKey(A)
    else:
        PressKey(W)
        ReleaseKey(A)
        ReleaseKey(D)
        print("W")
    k = output
    #output = [x for x in output[0]]
    #print(output)

    #output = [round(x) for x in output[0]]

    #cv2.imshow("window1", new1)
    #if cv2.waitKey(255) & 0xFF == ord('q'):
    #    cv2.destroyAllWindows()
    #    break
    time.sleep(0.4)'''

    '''if output[0][0] > output[0][1] and output[0][0] > output[0][2]:
        PressKey(A)
        ReleaseKey(W)
        ReleaseKey(D)
        print('A')
    elif output[0][2] > output[0][1] and output[0][2] > output[0][0]:
        PressKey(D)
        ReleaseKey(A)
        ReleaseKey(W)
        print('D')
    else:
        PressKey(W)
        ReleaseKey(A)
        ReleaseKey(D)
        print('W')'''

