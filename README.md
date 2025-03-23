# Lip Reading Model Demo

A Streamlit application that demonstrates lip reading using a deep learning model. The app allows users to select videos from the dataset, view predictions, and visualize the model's attention through heatmaps.

## Features

- Interactive video selection from the dataset
- Real-time video playback
- Lip reading predictions
- Heatmap visualization of model attention
- Side-by-side comparison of actual and predicted text

## Prerequisites

- Python 3.7+
- FFmpeg installed on your system
- TensorFlow 2.8.0 or higher
- Streamlit 1.22.0 or higher

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/lip-reading-demo.git
cd lip-reading-demo
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Install FFmpeg:
- Windows: `choco install ffmpeg`
- Ubuntu: `sudo apt-get install ffmpeg`
- macOS: `brew install ffmpeg`

## Project Structure

```
lip-reading-demo/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
├── models/            # Directory for saved models
│   └── model_epoch_30.h5
└── data/              # Dataset directory
    ├── s1/            # Video files
    └── alignments/    # Text alignment files
```

## Usage

1. Ensure your model is saved at `models/model_epoch_30.h5`
2. Place your dataset in the `data` directory with the following structure:
   - `data/s1/*.mpg` for video files
   - `data/alignments/s1/*.align` for alignment files

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LipNet paper: [LipNet: End-to-End Sentence-level Lipreading](https://arxiv.org/abs/1611.01599)
- Dataset: [GRID Corpus](http://spandh.dcs.shef.ac.uk/gridcorpus/) 