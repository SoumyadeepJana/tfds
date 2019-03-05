import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import os

_URL = "https://www.kaggle.com/moltean/fruits/downloads/fruits-360_dataset.zip/44"

_DESCRIPTION = ("The Fruits-360 dataset consists of 48905 training and 16412 testing images of size 100x100."
               "in 95 classes in the filename format image_index_100.jpg (e.g. 32_100.jpg) or"
               "r_image_index_100.jpg (e.g. r_32_100.jpg) or r2_image_index_100.jpg or r3_image_index_100.jpg." 
               "'r' stands for rotated fruit. 'r2' means that the fruit was rotated around the 3rd axis." 
               "'100' comes from image size (100x100 pixels).Different varieties of the same fruit (apple for instance) are stored as belonging to different classes.")

_FRUITS_IMAGE_SIZE = 100
_FRUITS_IMAGE_SHAPE = (_FRUITS_IMAGE_SIZE,_FRUITS_IMAGE_SIZE,3)

_URL1 = "https://www.kaggle.com/moltean/fruits/downloads/fruits-360_dataset.zip"
_URL2 = "https://github.com/Horea94/Fruit-Images-Dataset"
#https://www.kaggle.com/moltean/fruits/downloads/fruits-360_dataset.zip


_CITATION = """\
@article{murecsan2018fruit,
title={Fruit recognition from images using deep learning},
author={Mure{\c{s}}an, Horea and Oltean, Mihai},
journal={Acta Universitatis Sapientiae, Informatica},
volume={10},
number={1},
pages={26--42},
year={2018},
publisher={Sciendo}
}

"""

class Fruits(tfds.core.GeneratorBasedBuilder):
  """Fruits 360 dataset"""

  VERSION = tfds.core.Version('0.1.0')


  def _info(self):
     return tfds.core.DatasetInfo(
        builder=self,
  
        description=(_DESCRIPTION),
        
        features=tfds.features.FeaturesDict({
            "image": tfds.features.Image(shape=_FRUITS_IMAGE_SHAPE),
            "label": tfds.features.ClassLabel(num_classes=95),
        }),
        
        supervised_keys=("image", "label"),
        
        urls=[_URL1,_URL2],
        
        citation=_CITATION
     )

  def _split_generators(self, dl_manager):
    path = dl_manager.download_and_extract(_URL2)
    print("MOTHERUKCER:",path)
    return [
        tfds.core.SplitGenerator(
            name="train",
            num_shards=10,
            gen_kwargs={
                "images_dir_path": os.path.join(path, "Training")
                
            },
        ),
        tfds.core.SplitGenerator(
            name="test",
            num_shards=1,
            gen_kwargs={
                "images_dir_path": os.path.join(path, "Test"),
                
            },
        ),
    ]

  def _generate_examples(self,images_dir_path):
    for dirs in tf.io.gfile.listdir(images_dir_path):
      for d in dirs:
        folder = os.path.join(images_dir_path,d)
        for f in tf.io.gfile.listdir(folder):
           image_path = os.path.join(folder,f)
           yield {
                  "image": image_path,
                  "label": d.lower(),
              }
