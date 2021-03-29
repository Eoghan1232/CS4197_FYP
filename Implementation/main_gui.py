from tkinter import *
from tkinter import messagebox
from tkinter.ttk import *
from Doc2Vec import D2V
from NN import MovieDataset
from NN import ReutersDataset
from multiprocessing import Queue, Process


class UserGUI:

    def __init__(self):
        self.movie_dataset = MovieDataset()
        self.reuters_dataset = ReutersDataset()
        self.__root = Tk()
        self.__root.title("Document Classifier")
        self.__root.geometry('700x700')
        self.__root.resizable(True, False)
        self.__root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Tabs
        self.tab = Notebook(master=self.__root)
        self.tab.pack(fill=BOTH, expand=1, padx=15, pady=15)

        # Initiate Operations
        self.__init_movie_dataset_components()
        self.__init_reuters_dataset_components()

    def __init_movie_dataset_components(self):
        large_movie_review_dataset = Frame()
        large_movie_review_dataset.pack()
        self.tab.add(large_movie_review_dataset, text="Large Movie Review Dataset")

        # Top Frame
        top_frame = Frame(master=large_movie_review_dataset)
        top_frame.pack(fill=X, padx=15, pady=20)
        m = "The output throughout running the programme will be displayed in the Console Output window below. \n" \
            "Step1. Train the Doc2Vec model on the Movie_Review Dataset.\n" \
            "Step2. Train the CNN on the Doc2Vec model. You can also test the CNN on the test data.\n" \
            "Step3. You can input custom text into the input box and click Classify for the model to predict based on it. "
        message = Message(master=top_frame, text=m, relief="raised")
        message.bind("<Configure>", lambda event: event.widget.configure(width=event.width - 8))
        message.pack(fill=X)

        # Middle Frame
        training_frame = LabelFrame(master=large_movie_review_dataset, text="Training")
        training_frame.pack(fill=X, padx=15)
        middle_frame = Frame(master=training_frame)
        middle_frame.pack(fill=BOTH, padx=10, pady=5)

        # Construct the tree view
        self.treeview_movie = Treeview(master=middle_frame)
        self.treeview_movie.heading("#0", text="Console Output")
        self.treeview_movie.column("#0", anchor="center")
        self.treeview_movie.pack(fill=X, side=LEFT, expand=1)

        vsb = Scrollbar(master=middle_frame, orient="vertical", command=self.treeview_movie.yview)
        vsb.pack(side=RIGHT, fill=Y)
        self.treeview_movie.configure(yscrollcommand=vsb.set)

        button_frame = Frame(master=training_frame)
        button_frame.pack(padx=10, pady=10, fill=X, expand=0)
        button_in_frame = LabelFrame(master=button_frame, text="Step 1. Initialize & Train Doc2Vec model")
        button_in_frame.pack(side=LEFT, padx=10, pady=5)
        button_train_frame = LabelFrame(master=button_frame, text="Step 2. Train Classifier")
        button_train_frame.pack(side=RIGHT, padx=10, pady=5)

        self.train_doc2vec_model_movie = Button(master=button_in_frame, text="Train Doc2Vec model",
                                                command=self.train_d2v_movie)
        self.load_doc2vec_model_movie = Button(master=button_in_frame, text="Load Doc2Vec model",
                                               command=self.load_d2v_model_movie)
        self.train_classifier_movie = Button(master=button_train_frame, text="Train Classifier",
                                             command=self.train_model_movie)
        self.test_classifier_movie = Button(master=button_train_frame, text="Test Classifier",
                                            command=self.test_dataset_movie)

        self.train_doc2vec_model_movie.pack(side=LEFT, pady=5, padx=5)
        self.load_doc2vec_model_movie.pack(side=LEFT, pady=5, padx=5)
        self.train_classifier_movie.pack(side=LEFT, pady=5, padx=5)
        self.test_classifier_movie.pack(side=LEFT, pady=5, padx=5)

        # Bottom Frame
        bottom_frame = LabelFrame(master=large_movie_review_dataset, text="Step 3. Input Text for Classification")
        bottom_frame.pack(fill=X, padx=15, pady=15)

        self.input_text = Text(master=bottom_frame, height=5)
        self.input_text.pack(fill=X, padx=10, pady=10)

        classify_frame = Frame(master=bottom_frame)
        classify_frame.pack(padx=10, pady=10, fill=X, expand=0)

        self.classify_input = Button(master=classify_frame, text="Classify", command=self.classify_new_input_movie)
        self.classify_input.pack(side=LEFT)

    def __init_reuters_dataset_components(self):
        reuters_dataset = Frame()
        reuters_dataset.pack()
        self.tab.add(reuters_dataset, text="Reuters Dataset")

        top_frame = Frame(master=reuters_dataset)
        top_frame.pack(fill=X, padx=15, pady=20)
        m = "The output throughout running the programme will be displayed in the Console Output window below. \n" \
            "Step1. Train the Doc2Vec model on the Reuters Dataset.\n" \
            "Step2. Train the CNN on the Doc2Vec model. You can also test the CNN on the test data.\n" \
            "Finally. Testing the CNN will output the predicted vs. actual results on the screen to visually inspect."
        message = Message(master=top_frame, text=m, relief="raised")
        message.bind("<Configure>", lambda event: event.widget.configure(width=event.width - 8))
        message.pack(fill=X)

        # Middle Frame
        training_frame = LabelFrame(master=reuters_dataset, text="Training")
        training_frame.pack(fill=X, padx=15)
        middle_frame = Frame(master=training_frame)
        middle_frame.pack(fill=BOTH, padx=10, pady=5)

        # Construct the tree view
        self.treeview_reuters = Treeview(master=middle_frame)
        self.treeview_reuters.heading("#0", text="Console Output")
        self.treeview_reuters.column("#0", anchor="center")
        self.treeview_reuters.pack(fill=X, side=LEFT, expand=1)

        vsb = Scrollbar(master=middle_frame, orient="vertical", command=self.treeview_reuters.yview)
        vsb.pack(side=RIGHT, fill=Y)
        self.treeview_reuters.configure(yscrollcommand=vsb.set)

        # hsb = Scrollbar(master=training_frame, orient="horizontal", command=self.treeview_reuters.xview)
        # hsb.pack(side=TOP, fill=X)
        # self.treeview_reuters.configure(xscrollcommand=hsb.set)

        button_frame = Frame(master=training_frame)
        button_frame.pack(padx=10, pady=10, fill=X, expand=0)
        button_in_frame = LabelFrame(master=button_frame, text="Step 1. Initialize & Train Doc2Vec model")
        button_in_frame.pack(side=LEFT, padx=10, pady=5)
        button_train_frame = LabelFrame(master=button_frame, text="Step 2. Train Classifier")
        button_train_frame.pack(side=RIGHT, padx=10, pady=5)

        self.train_doc2vec_model_reuters = Button(master=button_in_frame, text="Train Doc2Vec model",
                                                  command=self.train_d2v_reuters)
        self.load_doc2vec_model_reuters = Button(master=button_in_frame, text="Load Doc2Vec model",
                                                 command=self.load_d2v_model_reuters)
        self.train_classifier_reuters = Button(master=button_train_frame, text="Train Classifier",
                                               command=self.train_model_reuters)
        self.test_classifier_reuters = Button(master=button_train_frame, text="Test Classifier",
                                              command=self.test_dataset_reuters)

        self.train_doc2vec_model_reuters.pack(side=LEFT, pady=5, padx=5)
        self.load_doc2vec_model_reuters.pack(side=LEFT, pady=5, padx=5)
        self.train_classifier_reuters.pack(side=LEFT, pady=5, padx=5)
        self.test_classifier_reuters.pack(side=LEFT, pady=5, padx=5)

    def train_model_movie(self):
        self.clear_output_movie()
        self.disable_buttons_movie()

        queue1 = Queue()

        process = Process(target=self.movie_dataset.train_dataset, args=(queue1,))
        process.start()

        self.__root.after(5000, self.check_result_movie, queue1)

    def train_model_reuters(self):
        self.clear_output_reuters()
        self.disable_buttons_reuters()

        queue1 = Queue()

        process = Process(target=self.reuters_dataset.train_dataset, args=(queue1,))
        process.start()

        self.__root.after(5000, self.check_result_reuters, queue1)

    def train_d2v_movie(self):
        self.clear_output_movie()
        self.disable_buttons_movie()

        queue1 = Queue()

        process = Process(target=self.movie_dataset.train_d2v, args=(queue1,))
        process.start()

        self.__root.after(5000, self.check_result_movie, queue1)
        # result = self.movie_dataset.train_dataset()
        # print(result)

    def train_d2v_reuters(self):
        self.clear_output_reuters()
        self.disable_buttons_reuters()

        queue1 = Queue()

        process = Process(target=self.reuters_dataset.train_d2v, args=(queue1,))
        process.start()

        self.__root.after(5000, self.check_result_reuters, queue1)

    def classify_new_input_movie(self):
        self.clear_output_movie()
        self.disable_buttons_movie()
        queue1 = Queue()

        input = self.input_text.get("1.0", "end-1c")

        process = Process(target=self.movie_dataset.test_dataset_new_input, args=(queue1, input))
        process.start()

        self.__root.after(5000, self.check_result_movie, queue1)

    def test_dataset_movie(self):
        self.clear_output_movie()
        self.disable_buttons_movie()
        queue1 = Queue()

        process = Process(target=self.movie_dataset.test_dataset, args=(queue1,))
        process.start()

        self.__root.after(5000, self.check_result_movie, queue1)

    def test_dataset_reuters(self):
        self.clear_output_reuters()
        self.disable_buttons_reuters()

        queue1 = Queue()

        process = Process(target=self.reuters_dataset.test_dataset, args=(queue1,))
        process.start()

        self.__root.after(5000, self.check_result_reuters, queue1)

    def load_d2v_model_movie(self):
        self.clear_output_movie()
        self.disable_buttons_movie()

        queue1 = Queue()

        process = Process(target=self.movie_dataset.load_d2v, args=(queue1, 1))
        process.start()

        self.__root.after(5000, self.check_result_movie, queue1)

    def load_d2v_model_reuters(self):
        self.clear_output_reuters()
        self.disable_buttons_reuters()

        queue1 = Queue()

        process = Process(target=self.reuters_dataset.load_d2v, args=(queue1, 1))
        process.start()

        self.__root.after(5000, self.check_result_reuters, queue1)

    def check_result_movie(self, param):
        print("method called")
        if param.empty():
            print("Queue empty")
            self.__root.after(5000, self.check_result_movie, param)
        else:
            # result == 0, not done. result == 1, done.
            callagain = True
            while not param.empty():
                result = param.get()
                if result[0] == 0:
                    self.update_text_movie(result[1])
                elif result[0] == 1:
                    self.update_text_movie(result[1])
                    self.enable_buttons_movie()
                    callagain = False
            if callagain:
                self.__root.after(5000, self.check_result_movie, param)
            # self.update_text(param.get()[0])
        # print("This is whats on the queue %s" % param.get())

    def check_result_reuters(self, param):
        print("method called")
        if param.empty():
            print("Queue empty")
            self.__root.after(5000, self.check_result_reuters, param)
        else:
            # result == 0, not done. result == 1, done.
            callagain = True
            while not param.empty():
                result = param.get()
                if result[0] == 0:
                    self.update_text_reuters(result[1])
                elif result[0] == 1:
                    self.update_text_reuters(result[1])
                    self.enable_buttons_reuters()
                    callagain = False
            if callagain:
                self.__root.after(5000, self.check_result_reuters, param)

    def update_text_movie(self, text):
        self.treeview_movie.insert("", "end", text=text)

    def update_text_reuters(self, text):
        self.treeview_reuters.insert("", "end", text=text)

    def enable_buttons_movie(self):
        self.train_doc2vec_model_movie['state'] = "normal"
        self.load_doc2vec_model_movie['state'] = "normal"
        self.classify_input['state'] = "normal"
        self.train_classifier_movie['state'] = "normal"
        self.test_classifier_movie['state'] = "normal"

    def enable_buttons_reuters(self):
        self.train_doc2vec_model_reuters['state'] = "normal"
        self.load_doc2vec_model_reuters['state'] = "normal"
        self.train_classifier_reuters['state'] = "normal"
        self.test_classifier_reuters['state'] = "normal"

    def disable_buttons_movie(self):
        self.train_doc2vec_model_movie['state'] = DISABLED
        self.load_doc2vec_model_movie['state'] = DISABLED
        self.classify_input['state'] = DISABLED
        self.train_classifier_movie['state'] = DISABLED
        self.test_classifier_movie['state'] = DISABLED

    def disable_buttons_reuters(self):
        self.train_doc2vec_model_reuters['state'] = DISABLED
        self.load_doc2vec_model_reuters['state'] = DISABLED
        self.train_classifier_reuters['state'] = DISABLED
        self.test_classifier_reuters['state'] = DISABLED

    def clear_output_movie(self):
        self.treeview_movie.delete(*self.treeview_movie.get_children())

    def clear_output_reuters(self):
        self.treeview_reuters.delete(*self.treeview_reuters.get_children())

    def display(self):
        self.__root.mainloop()

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.__root.destroy()
            sys.exit(0)
