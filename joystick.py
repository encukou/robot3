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
attrs = [0, 0, 0]
pressed_keys = set()

ROTATION_KEYS = {
    pyglet.window.key.Q: -50,
    pyglet.window.key.E: 50,
}
MOVEMENT_KEYS = {
    pyglet.window.key.W: (0, 1),
    pyglet.window.key.A: (-1, 0),
    pyglet.window.key.S: (0, -1),
    pyglet.window.key.D: (1, 0),
}

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
        label.text = str(speed)
        label.draw()
        speed = _clamp(speed)
        draw_rect(-MAX_SPEED/SHORTEN, SMALL_R-5, MAX_SPEED/SHORTEN, SMALL_R)
        gl.glColor4f(*color, 1)
        draw_rect(0, SMALL_R-5, speed/SHORTEN, SMALL_R)
        gl.glRotatef(360/3, 0, 0, 1)
    gl.glRotatef(math.degrees(attrs[0]), 0, 0, 1)
    gl.glColor4f(0.2, 0.2, 0.2, 1)
    draw_rect(0, -3, attrs[1]/SHORTEN, +3)
    gl.glPopMatrix()


@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    if buttons & pyglet.window.mouse.LEFT:
        set_xy(x, y)


@window.event
def on_mouse_press(x, y, buttons, modifiers):
    if buttons & pyglet.window.mouse.LEFT:
        set_xy(x, y)


@window.event
def on_mouse_release(x, y, buttons, modifiers):
    if buttons & pyglet.window.mouse.LEFT:
        set_xy(window.width/2, window.height/2)
        send_position()


@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    attrs[2] += scroll_y
    set_position()


@window.event
def on_key_press(keycode, mod):
    pressed_keys.add(keycode)
    handle_key(keycode)


@window.event
def on_key_release(keycode, mod):
    pressed_keys.discard(keycode)
    handle_key(keycode)


def handle_key(key):
    if key in ROTATION_KEYS:
        attrs[2] = 0
        for key, speed in ROTATION_KEYS.items():
            if key in pressed_keys:
                attrs[2] += speed
    if key in MOVEMENT_KEYS:
        x = y = 0
        angle = 0
        for key, (dx, dy) in MOVEMENT_KEYS.items():
            if key in pressed_keys:
                x += dx
                y += dy
        attrs[0] = math.atan2(y, x) + math.tau / 6
        attrs[1] = 1000 if y or x else 0
    set_position()


def set_xy(x, y):
    x -= window.width/2
    y -= window.height/2
    factor = min(window.width/SIZE, window.height/SIZE)
    x /= factor
    y /= factor
    distance = math.sqrt(x**2 + y**2) * MAX_SPEED / (BIG_R/2)
    if distance > 1000:
        distance = 1000
    angle = math.atan2(y, x) + math.tau/6
    attrs[:2] = angle, distance
    set_position()


def set_position():
    angle, distance, z = attrs
    for i in range(3):
        motor_angle = math.tau / 3 * i
        speeds[i] = int(math.cos(motor_angle - angle) * distance) + z * 10


def _clamp(speed):
    if speed < -1023:
        return -1023
    if speed > 1023:
        return 1023
    return speed


def send_position(dt=None):
    message = json.dumps([_clamp(s) for s in speeds]).encode() + b'\n'
    sock.sendall(message)

pyglet.clock.schedule_interval(send_position, 1/10)

pyglet.app.run()
