# Image Classification model VGG16 from scratch | Computer Vision with Pytorch --- in devloper



<p align="center">
<img src="./fig/ArchitectureVgg16.png" width="500px"></img>
</p>



<p align="center">
<img src="./fig/image-33.png" width="500px"></img>
</p>



```python

print('==> Building model..')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = VGG('VGG11',n_classe).to(device)

print(net)

net = net.to(device)

print(device)

```
