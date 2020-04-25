from tkinter import *
from tkinter.colorchooser import askcolor
from PIL import ImageGrab
import model


class Paint(object):

	DEFAULT_COLOR = 'black'

	def __init__(self):
		self.root = Tk()

		# Button to enable the brush. Enabled by default
		self.brush_button = Button(self.root, text='brush', command=self.use_brush)
		self.brush_button.grid(row=0, column=0)

		# Button that allows you to change color
		self.color_button = Button(self.root, text='color', command=self.choose_color)
		self.color_button.grid(row=0, column=1)

		# Button that resets the canvas
		self.reset_button = Button(self.root, text='reset', command=self.reset)
		self.reset_button.grid(row=0, column=2)

		# Button to enable the eraser
		self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
		self.eraser_button.grid(row=0, column=3)

		# Scale to change brush size from 20-50
		self.choose_size_button = Scale(self.root, from_=20, to=50, orient=HORIZONTAL)
		self.choose_size_button.grid(row=0, column=4)

		# Button to start prediction by calling the model from model.py
		self.predict = Button(self.root, text='Predict', command=self.predict)
		self.predict.grid(row=2, column=3)

		# Textfield that outputs the prediction
		self.answer = Text(self.root, height=2, width=10, font=("Helvetica", 32))
		self.answer.tag_configure("center", justify='center')
		self.answer.insert(INSERT, " ")
		self.answer.grid(row=3, column=3)

		# Canvas used to draw the character
		self.c = Canvas(self.root, bg='white', width=200, height=200, highlightthickness=1, highlightbackground="black")
		self.c.grid(row=1, columnspan=6)

		self.setup()
		self.root.mainloop()

	def setup(self):
		self.old_x = None
		self.old_y = None
		self.line_width = self.choose_size_button.get()
		self.color = self.DEFAULT_COLOR
		self.eraser_on = False
		self.active_button = self.brush_button
		self.c.bind('<B1-Motion>', self.paint)
		self.c.bind('<ButtonRelease-1>', self.pick_up)

	def use_brush(self):
		self.eraser_on = False
		self.activate_button(self.brush_button)

	def choose_color(self):
		self.eraser_on = False
		self.color = askcolor(color=self.color)[1]

	def use_eraser(self):
		self.activate_button(self.eraser_button, eraser_mode=True)

	def predict(self):
		# First get the dimension of the image
		x = self.c.winfo_rootx() + self.c.winfo_x() + 10
		y = self.c.winfo_rooty() + self.c.winfo_y() + 57
		x1 = x + self.c.winfo_width() - 7
		y1 = y + self.c.winfo_height() - 20

		# Now crop the image to only receive the canvas. The 200 is because of the dimension of the canvas
		ImageGrab.grab().crop((x, y, x1 + 200, y1 + 200)).save("my_drawing.png")
		self.ans()

	def ans(self):
		# Will call the model in model.py
		prediction = model.main()
		self.answer.delete('1.0', END)
		self.answer.insert(INSERT, str(prediction))
		self.answer.tag_add("center", "1.0", "end")
		pass

	def activate_button(self, new_button, eraser_mode=False):
		self.active_button.config(relief=RAISED)
		new_button.config(relief=SUNKEN)
		self.active_button = new_button
		self.eraser_on = eraser_mode

	def paint(self, event):
		self.line_width = self.choose_size_button.get()
		paint_color = 'white' if self.eraser_on else self.color
		if self.old_x and self.old_y:
			self.c.create_line(self.old_x, self.old_y, event.x, event.y,
							   width=self.line_width, fill=paint_color,
							   capstyle=ROUND, smooth=False, splinesteps=36)
		self.old_x = event.x
		self.old_y = event.y

	# Pick up the brush
	def pick_up(self, event):
		self.old_x, self.old_y = None, None

	# Resets the canvas
	def reset(self):
		self.c.delete('all')


if __name__ == '__main__':
	Paint()