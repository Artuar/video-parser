from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads
from src.index import get_photos_from_video
from src.utils import remove_file
from src.path import VIDEO_STREAM

ALLOWED_EXTENSIONS = {'mp4'}
FILE_NAME = "current.mp4"

app = Flask(__name__)
photos = UploadSet('photos', ALLOWED_EXTENSIONS)
app.config['UPLOADED_PHOTOS_DEST'] = 'videos'
configure_uploads(app, photos)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'video' in request.files:
        remove_file(VIDEO_STREAM, FILE_NAME)
        file = request.files['video']
        file.filename = FILE_NAME
        filename = photos.save(request.files['video'])
        links = get_photos_from_video(filename)
        return render_template('upload.html', photos=links)
    return render_template('upload.html', photos=[])


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
