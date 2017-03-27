import asyncio
import json

import pandas as pd
import websockets


def format_data(data_frame):
    timestamp = int(float(data_frame.columns[0]))
    data_frame.columns = ['data']

    fs = data_frame.data[0]
    data_frame = data_frame.drop(
        range(5 * int(fs)))  # drop sample rate and first 5 seconds due to instrument calibration
    data = data_frame.data.as_matrix()

    return timestamp, fs, data


def get_data(directory):
    return format_data(pd.read_csv(directory + '/BVP.csv'))


def grouped(iterable, n):
    return zip(*[iter(iterable)] * n)


async def analyze_live(bvp, chunk_size):
    async with websockets.connect('ws://127.0.0.1:5000') as ws:
        await ws.send(json.dumps({'command': 'restart', 'args': ''}))
        await ws.send(json.dumps({'command': 'change_mode', 'args': {'mode': 'EffortDerivative'}}))
        for data in grouped(bvp, chunk_size):
            data_json = json.dumps(data)
            await ws.send(data_json)


if __name__ == '__main__':
    timestamp, fs, bvp = get_data('/Users/francois/GoogleDrive/Work/Carleton/4000/SYSC4907/1488670526_A0031B')
    asyncio.get_event_loop().run_until_complete(analyze_live(bvp, 32))
