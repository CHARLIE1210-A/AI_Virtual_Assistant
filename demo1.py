from pydub import AudioSegment
from pydub.playback import play

audio = AudioSegment.from_file("output.mp3", format="mp3")
play(audio)