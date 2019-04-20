import network
import socket
from time import sleep
from machine import Pin, PWM

class Motor:
    def __init__(self, a, b, en):
        self.a = Pin(a, Pin.OUT)
        self.b = Pin(b, Pin.OUT)
        self.pwm = PWM(Pin(en, Pin.OUT), freq=100, duty=0)

    def __call__(self, value):
        if value < 0:
            value = -value
            self.a(0)
            self.b(1)
        else:
            self.a(1)
            self.b(0)
        self.pwm.duty(value)

motors = [
    Motor(21, 22, 23),
    Motor(5, 18, 19),
    Motor(2, 15, 4),
]



wlan = network.WLAN(network.STA_IF)
wlan.active(False)
wlan.active(True)
wlan.connect('ESSID', 'PWD') # connect to an AP
while not wlan.isconnected():
    print('connecting...')
    sleep(1)
print(wlan.ifconfig())


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('0.0.0.0', 29842)
sock.bind(server_address)
sock.listen(1)

while True:
    connection, client_address = sock.accept()
    print('connected to', client_address)
    connection.setblocking(0)

    data = ''
    while True:
        try:
            received = connection.recv(32).decode()
            if not received:
                print('disconnected')
                break
        except OSError:
            pass
        else:
            print('got', repr(received))
            data += received
        while '\n' in data:
            command, sep, data = data.partition('\n')
            print(command)
        sleep(1/10)
