import json

import numpy as np
import os
from django.shortcuts import render

session_path = 'BreathingEffortAnalysisServer' + os.sep + 'saved'


# Create your views here.
def get_sessions():
    return os.listdir(session_path)


def get_session_prefix(session_id):
    return session_path + os.sep + session_id + os.sep + session_id


def open_session(session_id):
    session_prefix = get_session_prefix(session_id)
    f_samples_name = session_prefix + '-samples.npy'
    f_effort_name = session_prefix + '-effort.npy'
    with open(f_samples_name, 'rb') as f_samples, open(f_effort_name, 'rb') as f_effort:
        samples = np.fromfile(f_samples)
        effort = np.fromfile(f_effort)
        return samples, effort


def index(request):
    analysis_sessions = get_sessions()
    return render(request, 'index.html', context={'sessions': analysis_sessions})


def session(request, id):
    samples, effort = open_session(id)
    return render(request, 'session.html',
                  context={'samples': json.dumps(samples.tolist()), 'effort': json.dumps(effort.tolist())})
