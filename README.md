# Image-classification-via-brain-like-learning-in-neural-networks

In this project, I compare the differences between SNN and ANN in terms of accuracy and loss and give my own explaination to the facts that I observed. 

This repository contains the file of the project. Before running the notebook, you need to add some code to grad-cam's package. 

Python\Python310\lib\site-packages\pytorch_grad_cam\base_cam.py

at line 85

'''

####################################################
'''
if targets is None:
    target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
   targets = [ClassifierOutputTarget(
        category) for category in target_categories]
        '''
####################################################
'''
add the code in forward() below 'if targets is None:'

if len(outputs)>1:
    outputs=outputs[0]

Then it looks like:

'''
####################################################
'''
if targets is None:
    #SNN has 2 outputs, spk_out and mem_out. Here we take the first output. 
    if len(outputs)>1:
        outputs=outputs[0]
    target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
   targets = [ClassifierOutputTarget(
        category) for category in target_categories]
'''
####################################################
