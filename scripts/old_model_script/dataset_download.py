from roboflow import Roboflow
rf = Roboflow(api_key="SkrKXksRVrJbBrYi1Ifc")
project = rf.workspace("yami-dzhfx").project("construction-ppe-qw7qk")
version = project.version(1)
dataset = version.download("paligemma")