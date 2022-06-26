from flask import Flask,request,render_template, redirect, url_for, flash
from matplotlib.pyplot import fill
from solve_sudoku import build_sudoku_matrix, sudoku_result
import os
from werkzeug.utils import secure_filename

app = Flask(__name__, static_url_path='/static')
app.secret_key = 'super secret'

app.config["IMAGE_UPLOADS"] = ".\static"
allowed_extensions = ['jpg', 'png', 'jpeg']

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/upload_sudoku',methods = ["GET","POST"])
def upload_sudoku():
	if request.method == "POST":
		image = request.files['file']

		if image.filename == '':
			print("Image must have a file name")
			return redirect(request.url)

		filename = secure_filename(image.filename)

		if not allowed_file(filename):
			flash("You can only upload an image file.", 'danger')
			return redirect('/')

		basedir = os.path.abspath(os.path.dirname(__file__))
		image.save(os.path.join(basedir,app.config["IMAGE_UPLOADS"],filename))

		grid = build_sudoku_matrix(filename)
		os.remove(os.path.join(basedir,app.config["IMAGE_UPLOADS"],filename))

		grid = sudoku_result(grid)

		if grid == -1:
			flash("Oops! There might be some problem with the uploaded sudoku image. Try uploading a different image or enter the sudoku values manually.", "warning")
			return redirect('/')
		else:
			return render_template("sudoku_result.html", grid=grid)

	return render_template('upload_sudoku.html')


# @app.route('/sudoku_result', methods = ["GET","POST"])
# def sudoku_result(filename):
# 	grid = build_sudoku_matrix(filename)

# 	return render_template("sudoku_result.html", filename=filename, grid=grid)


@app.route('/enter_sudoku',methods = ["GET","POST"])
def enter_sudoku():
	if request.method == "POST":
		grid = [[0 for x in range(9)]for y in range(9)]

		for row in range(0,9):
			for col in range(0,9):
				val = request.form[f"{row}{col}"]
				if val == '':
					grid[row][col] = 0
				else:
					grid[row][col] = int(val)

		grid = sudoku_result(grid)

		if grid == -1:
			flash("Sudoku cannot be solved with the current entered values. Please check the values you entered and try again.", "warning")
			return redirect('/')
		else:
			return render_template("sudoku_result.html", grid=grid)

	return render_template("enter_sudoku.html")


app.run(debug=True)