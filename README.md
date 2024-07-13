
## Download the FakeBr dataset:

  
Download the dataset from [FakeBr GitHub](https://github.com/roneysco/Fake.br-Corpus) and place the fake and true directories inside a directory named texts in the project root.  
  
	penalty-kick-prediction/
	│
	├── texts/
	│   ├── fake/
	│   │   ├── fake1.txt
	│   │   ├── fake2.txt
	│   │   └── ...
	│   └── true/
	│       ├── true1.txt
	│       ├── true2.txt
	│       └── ...
	│
	├── .venv/
	│   └── ...
	│
	├── main.py
	├── requirements.txt
	└── README.md


    python main.py  

  - texts/: Directory containing the FakeBr dataset.    main.py: Main
  -  script for processing the text data and training the machine learning
  - model.    requirements.txt: List of required Python packages.

  

## Results

- Accuracy of the model: 95%  
- Number of words/bigrams/trigrams used: 5000  
- Pre-processing techniques: Conversion to lowercase, removal of punctuation, tokenization, removal of stopwords.  
- Model chosen: Logistic Regression