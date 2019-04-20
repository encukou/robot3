import math
import socket
import json
import time

import pyglet
from pyglet import gl

SIZE = 700
BIG_R = 500
SMALL_R = 150
MAX_SPEED = 1000
SHORTEN = 4

window = pyglet.window.Window(width=SIZE,height=SIZE, resizable=True)
speeds = [0, 0, 0]
attrs = [0, 0]

label = pyglet.text.Label('0', y=SMALL_R)


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('192.168.0.46', 29842)
server_address = ('192.168.0.36', 29842)
sock.connect(server_address)


def draw_rect(x1, y1, x2, y2):
    gl.glBegin(gl.GL_TRIANGLE_FAN)
    gl.glVertex2f(int(x1), int(y1))
    gl.glVertex2f(int(x1), int(y2))
    gl.glVertex2f(int(x2), int(y2))
    gl.glVertex2f(int(x2), int(y1))
    gl.glEnd()


def draw_circle(x, y, radius):
    iterations = 50
    s = math.sin(2*math.pi / iterations)
    c = math.cos(2*math.pi / iterations)

    dx, dy = radius, 0

    gl.glBegin(gl.GL_LINE_STRIP)
    for i in range(iterations+1):
        gl.glVertex2f(x+dx, y+dy)
        dx, dy = (dx*c - dy*s), (dy*c + dx*s)
    gl.glEnd()


@window.event
def on_draw():
    window.clear()
    gl.glColor4f(1, 1, 1, 1)
    gl.glPushMatrix()
    gl.glTranslatef(window.width/2, window.height/2, 0)
    factor = min(window.width/SIZE, window.height/SIZE)
    gl.glScalef(factor, factor, 1)
    gl.glRotatef(-60, 0, 0, 1)
    draw_circle(0, 0, BIG_R/2)
    draw_circle(0, 0, SMALL_R)
    draw_circle(0, 0, 20)
    for speed, color in zip(speeds, [(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
        gl.glColor4f(0.2, 0.2, 0.2, 1)
        draw_rect(-MAX_SPEED/SHORTEN, SMALL_R-5, MAX_SPEED/SHORTEN, SMALL_R)
        label.text = str(speed)
        label.draw()
        gl.glColor4f(*color, 1)
        draw_rect(0, SMALL_R-5, speed/SHORTEN, SMALL_R)
        gl.glRotatef(360/3, 0, 0, 1)
    gl.glRotatef(math.degrees(attrs[0]), 0, 0, 1)
    gl.glColor4f(0.2, 0.2, 0.2, 1)
    draw_rect(0, -3, attrs[1]/SHORTEN, +3)
    gl.glPopMatrix()


@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    send(x, y)


@window.event
def on_mouse_press(x, y, buttons, modifiers):
    send(x, y)


@window.event
def on_mouse_release(x, y, buttons, modifiers):
    send(window.width/2, window.height/2)
    send_position()


def send(x, y):
    x -= window.width/2
    y -= window.height/2
    factor = min(window.width/SIZE, window.height/SIZE)
    x /= factor
    y /= factor
    distance = math.sqrt(x**2 + y**2) * MAX_SPEED / (BIG_R/2)
    if distance > 1000:
        distance = 1000
    angle = math.atan2(y, x) + math.tau/6
    speeds[0] = distance
    for i in range(3):
        motor_angle = math.tau / 3 * i
        speeds[i] = int(math.cos(motor_angle - angle) * distance)
    attrs[:] = angle, distance


def send_position(dt=None):
    sock.sendall(json.dumps(speeds).encode() + b'\n')

pyglet.clock.schedule_interval(send_position, 1/10)

pyglet.app.run()
