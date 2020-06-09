# Simple telegram style transfer BOT
Send photo to this guy and you hopefully will see some anime style :)

Thanks to Yijunmaverick for his simple inference [implementation](https://github.com/Yijunmaverick/CartoonGAN-Test-Pytorch-Torch)

## Dependencides
Code tested under:
* Ubuntu 18.04
* Cuda 10.2 cudnn 7.6.5
* Pytorch 1.5

```sh
pip3 install -r requirements.txt
```

Get weights for CartoonGan
```sh
sudo chmod +x get_weights.sh
./get_weights.sh
```

## Credentials
Creqte config file 
```sh
touch config.py
```
And fill your bot token and your proxy
```python
from config_base import _Config

class Config(_Config):
    TOKEN = 'your_token'
    REQUEST_KWARGS = {'proxy_url': 'http://some_proxy_ip:some_proxy_port/'}
```

## Run
Simple part
```sh
python3 main.py
```
