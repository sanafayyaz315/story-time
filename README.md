# story-time
In my Story-Time repo, I personally built and pretrained a GPT-2 Small story generator—complete with training scripts and a FastAPI inference API that you can clone and run locally to turn any prompt into a creative narrative.

##---------------README TO BE IMPROVED-----------------##

For now, to run inference, 

1. Clone the repository
``` 
git clone https://github.com/sanafayyaz315/story-time
```

2. Download the requirements using 
```
pip install -r requirements.txt
```
3. Download weights from Google Drive link `https://drive.google.com/file/d/1JZ7qipP-0NpZQOA8v7bmsIzM6iCLGvR_/view?usp=sharing`
and save in the ```checkpoints``` folder

4. Run the FastAPI application
``` 
cd src
python app.py
```

5. Run the UI
```
streamlit run streamlit_ui.py
```