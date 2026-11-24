# Data

`raw/Fake.csv` (23,481 articles) and `raw/True.csv` (21,417 articles) come from the Kaggle
dataset **Fake and Real News Dataset** (Clément Bisaillon):
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Columns: `title`, `text`, `subject`, `date`. Labels are assigned by the scripts
(0 = fake, 1 = real). Please check the Kaggle page for the dataset's license and
terms before redistributing.

Note: `Fake.csv` is ~63 MB. GitHub warns above 50 MB and blocks files above 100 MB.
If you prefer not to commit the data, delete `data/raw/*.csv` and add them to `.gitignore`,
or track them with Git LFS: `git lfs track "data/raw/*.csv"`.

Known quirks: 3 duplicate rows in Fake.csv and 206 in True.csv; most real articles begin with
a "CITY (Reuters) -" dateline, which leaks the label (see the main README).
