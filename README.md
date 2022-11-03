# Image Classification model VGG16 from scratch | Computer Vision with Pytorch --- in devloper



<p align="center">
<img src="./fig/ArchitectureVgg16.png" width="500px"></img>
</p>



<p align="center">
<img src="./fig/image-33.png" width="500px"></img>
</p>

bloco 1 


```python

class vgg16scratch(nn.Module):
    def __init__(self):
        super(vgg16scratch, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=(2,2)), # pool de janela quadrada de tamanho = 2, passo = 2
        )
    def forward(self, x):
        x = self.layer1(x)
        return x

```
