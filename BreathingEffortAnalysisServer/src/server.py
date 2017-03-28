import asyncio
import json

import matplotlib as mpl

mpl.use('tkagg')

import matplotlib.pyplot as plt
import websockets
from websockets.exceptions import ConnectionClosed
import time

from effort import *

win_secs = 15
fs = 64
winSize = win_secs * fs
q = EffortQueue(winSize, EffortAmplitudeModulation())
plot_bvp = None
plot_effort = None


async def received_data(websocket, path):
    timestring = time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime())
    f_samples_name = '../saved/' + timestring + '-samples.npy'
    f_effort_name = '../saved/' + timestring + '-effort.npy'
    with open(f_samples_name, 'a+b') as f_samples, open(f_effort_name, 'a+b') as f_effort:
        while True:
            try:
                await handle_received_data(websocket, f_samples, f_effort)
            except ConnectionClosed:
                print('Connection closed')
                break
        print('Uploading files')
        f_samples.seek(0)
        f_effort.seek(0)


@asyncio.coroutine
def refresh_plot():
    while True:
        plt.pause(1 / 60)
        yield


async def handle_received_data(websocket, f_samples, f_effort):
    message = await websocket.recv()
    try:
        message_json = json.loads(message)
        if 'command' in message_json:
            response = run_command(message_json)
            print(response)
            if response:
                await websocket.send(json.dumps(response))
        else:
            new_samples = np.asarray(message_json)
            latest_effort = q.feed(new_samples)

            f_samples.write(new_samples.tobytes())
            f_effort.write(latest_effort)

            plot_bvp_and_effort()
    except ValueError as e:
        print(e)


def plot_bvp_and_effort():
    effort = q.effort()
    plot_effort.clear()
    plot_effort.plot(effort)
    plot_effort.title.set_text(q.calculator.__class__.__name__)
    plot_effort.set_xlim([0, len(effort)])

    bvp = q.samples()
    plot_bvp.clear()
    plot_bvp.plot(bvp)
    plot_bvp.title.set_text('BVP')
    plot_bvp.set_xlim([0, len(bvp)])
    plot_bvp.set_ylim([-150, 150])

    plt.pause(1 / 60)


def run_command(message_json):
    try:
        command = message_json['command']
        args = message_json['args']
        print('running command: {}({})'.format(command, args))
        if command == 'restart':
            restart()
        elif command == 'change_mode':
            mode = args['mode']
            change_mode(mode)
        elif command == 'list_modes':
            modes = [k for k in globals().keys()
                     if k.startswith('Effort')
                     and not k == 'EffortCalculator'
                     and not k == 'EffortQueue']
            return {'modes': sorted(modes)}
    except KeyError as e:
        return json.dumps({'error': repr(e)})


def restart():
    print('restarting server')
    plot_bvp.clear()
    plot_effort.clear()
    plt.pause(0.001)
    q.clear()


def change_mode(mode: str):
    q.calculator = globals()[mode]()
    print('changed effort calculation mode to ' + mode)
    restart()


if __name__ == "__main__":
    plt.ioff()
    _, [plot_bvp, plot_effort] = plt.subplots(2, 1)

    server = websockets.serve(received_data, '0.0.0.0', 5000)
    asyncio.get_event_loop().run_until_complete(server)
    # asyncio.ensure_future(refresh_plot())
    asyncio.get_event_loop().run_forever()
