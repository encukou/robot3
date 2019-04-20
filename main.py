import network
import socket
from time import sleep
from machine import Pin, PWM
import uasyncio as asyncio
import json

from wificonfig import wifi_essid, wifi_password

class Motor:
    def __init__(self, a, b, en):
        self.a = Pin(a, Pin.OUT)
        self.b = Pin(b, Pin.OUT)
        self.pwm = PWM(Pin(en, Pin.OUT), freq=100, duty=0)

    def __call__(self, value):
        if value == 0:
            # Brake
            self.a(0)
            self.b(0)
            self.pwm.duty(1)
        else:
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
wlan.connect(wifi_essid, wifi_password) # connect to an AP
while not wlan.isconnected():
    print('connecting...')
    sleep(1)
print(wlan.ifconfig())


async def serve(reader, writer):
    print('connected')
    await writer.awrite('"hello"\n')
    while True:
        try:
            line = (await reader.readline()).decode()
        except OSError:
            continue
        if line == '':
            break
        print(line.strip())
        try:
            command = json.loads(line)
        except ValueError:
            print('bad value')
        else:
            print('command', command)
            if isinstance(command, list):
                for motor, speed in zip(motors, command):
                    print('setting', motor, 'to', speed)
                    motor(speed)
    print('disconnected')
    await writer.aclose()

async def reset_loop():
    boot_pin = Pin(0, Pin.IN)
    while True:
        if not boot_pin():
            for motor in motors:
                motor(0)
        await asyncio.sleep_ms(50)


loop = asyncio.get_event_loop()
loop.call_soon(asyncio.start_server(serve, "0.0.0.0", 29842))
loop.run_forever()
loop.close()
