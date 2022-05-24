from contextlib import contextmanager
from io import StringIO
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from threading import current_thread
import streamlit as st
from detect import main
import os
import sys
import argparse
from PIL import Image
from pathlib import Path

st.set_page_config(page_title='Computer Vision | Demo Hub', layout='wide')


@contextmanager
def st_redirect(src, dst):
    """
    Redirects the print of a function to the streamlit UI
    """
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst):
    """
    Sub-implementation to redirect for code readability.
    """
    with st_redirect(sys.stdout, dst):
        yield


@contextmanager
def st_stderr(dst):
    """
    Sub-implementation to redirect for code readability in case of errors.
    """
    with st_redirect(sys.stderr, dst):
        yield


def _all_subdirs_of(b='.'):
    """
    Returns all sub-directories in a specific Path
    """
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result


def _get_latest_folder():
    """
    Returns the latest folder in a runs/detect
    """
    return max(_all_subdirs_of(os.path.join('runs', 'detect')), key=os.path.getmtime)


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory

parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='inference size h,w')
parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='show results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--visualize', action='store_true', help='visualize features')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
opt = parser.parse_args()


def _save_uploadedfile(uploadedfile):
    """
    Saves uploaded videos to disk
    """
    with open(os.path.join('data', 'videos', uploadedfile.name), 'wb') as f:
        f.write(uploadedfile.getbuffer())


def _sources_func(option):
    """
    Format function for select Key/Value implementation.
    """
    return SOURCE_CHOICES[option]


def _preload_func(option):
    """
    Format function for select Key/Value implementation.
    """
    return PRELOAD_CHOICES[option]


# --- App Start --- #

title = st.title('Computer Vision | Demo Hub')
subheader = st.text('created by the Capgemini Computer Vision Guild (UK I&D)')

blank_space_0 = st.text('')

problem_description = st.markdown(
    '**Object detection is the task of detecting instances of objects of a certain class '
    'within an image.** Object detection has applications in many areas of computer vision, '
    'including image annotation, vehicle counting, activity recognition, face detection, '
    'face recognition, video object co-segmentation.')

select_problem = st.sidebar.selectbox('Select problem type:',
                                      ('Object Detection', 'Object Segmentation', 'Style Transfer', 'Image Synthesis'))

blank_space_1 = st.text('')

a1, a2 = st.columns(2)

with a1:
    select_model = st.selectbox('Model & Architecture:',
                                ('YOLOv5', 'YOLOv4', 'etc.'))

    blank_space_2 = st.text('')

with a2:
    select_training = st.selectbox('Training Data & Weights:',
                                   ('Parking Lots', 'Cats & Dogs', 'etc.'))

    blank_space_3 = st.text('')

b1, b2, b3, b4 = st.columns([1, 3, 1, 3])

with b1:
    st.image('data/images/bus.jpg')

with b2:
    model_description = st.write('Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor '
                                 'incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud '
                                 'exercitation ullamco laboris nisi ut aliquip ex ea commodo')

with b3:
    st.image('data/images/bus.jpg')

with b4:
    training_description = st.write('Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor '
                                    'incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, '
                                    'quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo')

blank_space_4 = st.text('')

c1, c2 = st.columns(2)

SOURCE_CHOICES = {0: 'Image Upload', 1: 'Video Upload', 2: 'Pre-loaded Options'}
PRELOAD_CHOICES = {0: 'Bus', 1: 'Zidane', 2: 'Video option'}
PRELOAD_PATHS = {0: 'data/images/bus.jpg', 1: 'data/images/zidane.jpg', 2: 'data/images/bus.jpg'}

with c1:
    inferenceSource = str(st.radio('Select source:',
                                   options=list(SOURCE_CHOICES.keys()),
                                   format_func=_sources_func))
with c2:
    if inferenceSource == '0':
        uploaded_file = st.file_uploader('Upload image', type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='In progresss'):
                st.sidebar.text('Uploaded File Preview:')
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                picture = picture.save(f'data/images/{uploaded_file.name}')
                opt.source = f'data/images/{uploaded_file.name}'
        else:
            is_valid = False
    elif inferenceSource == '1':
        uploaded_file = st.file_uploader('Upload video', type=['mp4'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='In progress'):
                st.video(uploaded_file)
                _save_uploadedfile(uploaded_file)
                opt.source = f'data/videos/{uploaded_file.name}'
        else:
            is_valid = False
    elif inferenceSource == '2':
        pre_loaded_selection = st.radio('Select a pre-loaded image:',
                                        options=list(PRELOAD_CHOICES.keys()),
                                        format_func=_preload_func)
        is_valid = True
        uploaded_file = PRELOAD_PATHS[pre_loaded_selection]
        with st.spinner(text='In progresss'):
            st.sidebar.text('Uploaded File Preview:')
            st.sidebar.image(uploaded_file)
            opt.source = str(uploaded_file)

inferenceButton = st.empty()

if is_valid:
    if inferenceButton.button('Begin inference'):
        with st_stdout('info'):
            main(opt)
        if inferenceSource == '1':
            st.warning('Video playback not available')
            with st.spinner(text='Preparing Video'):
                for vid in os.listdir(_get_latest_folder()):
                    st.video(f'{_get_latest_folder()}/{vid}')
        else:
            with st.spinner(text='Preparing Image'):
                for img in os.listdir(_get_latest_folder()):
                    st.image(f'{_get_latest_folder()}/{img}', use_column_width='auto')
