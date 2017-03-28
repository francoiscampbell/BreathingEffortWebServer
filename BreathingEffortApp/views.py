import json

import numpy as np
import os
from django.shortcuts import render

session_path = 'BreathingEffortAnalysisServer' + os.sep + 'saved'


# Create your views here.
def get_sessions():
    return os.listdir(session_path)


def get_session_prefix(session_id: str):
    session_id = session_id.replace('..', '')
    return session_path + os.sep + session_id + os.sep + session_id


def open_session(session_id):
    session_prefix = get_session_prefix(session_id)
    f_samples = session_prefix + '-samples.npy'
    f_effort = session_prefix + '-effort.npy'
    return np.fromfile(f_samples), np.fromfile(f_effort)


def index(request):
    analysis_sessions = get_sessions()
    return render(request, 'index.html', context={'sessions': analysis_sessions})


def session(request, session_id):
    samples, effort = open_session(session_id)
    canvas_width_vw = len(samples) / 64 / 15 * 100
    return render(request, 'session.html',
                  context={
                      'samples': json.dumps(samples.tolist()),
                      'effort': json.dumps(effort.tolist()),
                      'canvas_width': min(canvas_width_vw, 1000)
                  })
