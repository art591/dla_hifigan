# Implementation of HIFI-GAN

Implementation of [HIFI-GAN](https://arxiv.org/abs/2010.05646) for the "Deep Learning in Audio" course.

1. Install all needed packages with ```pip install -r requirements.txt``` .
2. Download LJSpeech dataset with ```./loading_data.sh```.
3. Download generator checkpoint with ```./loading_generator.sh```.
4. Put true audio into the ```test_data/true``` folder.
5. Run ```test.py``` to get predictions in ```test_data/pred```.

Some example test audio and corresponding predictions are already in ```test_data```.

