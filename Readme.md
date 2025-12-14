如果遇到
```
==================== 测试 PG-19 (单本) ====================
trust_remote_code is not supported anymore.
Please check that the Hugging Face dataset 'deepmind/pg19' isn't based on a loading script and remove trust_remote_code.
If the dataset is based on a loading script, please ask the dataset author to remove it and convert it to a standard format like Parquet.
Traceback (most recent call last):
File "e:\Development\NLP-FinalLab\baseline.py", line 108, in <module>
pg19_stream = load_dataset(
File "C:\Users\17134\miniconda3\envs\nlp\lib\site-packages\datasets\load.py", line 1397, in load_dataset
builder_instance = load_dataset_builder(
File "C:\Users\17134\miniconda3\envs\nlp\lib\site-packages\datasets\load.py", line 1137, in load_dataset_builder
dataset_module = dataset_module_factory(
File "C:\Users\17134\miniconda3\envs\nlp\lib\site-packages\datasets\load.py", line 1036, in dataset_module_factory
raise e1 from None
File "C:\Users\17134\miniconda3\envs\nlp\lib\site-packages\datasets\load.py", line 994, in dataset_module_factory
raise RuntimeError(f"Dataset scripts are no longer supported, but found {filename}")
RuntimeError: Dataset scripts are no longer supported, but found pg19.py
```
的问题，这可能是因为你的 datasets 库版本过高。

请尝试降级你的 datasets 库到 3.x.x版本，例如 `pip install datasets=3.6.0`