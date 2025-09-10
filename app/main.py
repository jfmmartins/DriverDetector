from tensorflow.keras.models import load_model
from ml import train
from webcam import web_video



def main():
    try: 
        print("Loading model")
        model = load_model("models/eye_state_cnn.h5")
        print("Model Loaded")
    except Exception:
        print("No model found in the folder\n")
        print("Training started\n")
        train()
        model = load_model("models/eye_state_cnn.h5")


    web_video(model)     



if __name__ == "__main__":
    main()
