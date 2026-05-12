https://github.com/5CCSACCA/coursework-51b5a.git 

The purpose of this project is to learn how to build an architecture that connects two models: an image classification model (YOLO) and an LLM (BitNet). 
This project takes that concept and creates a musuem tour guide that takes an artifact image as input and returns an llm output that describes the artifact
using the department and object that was classified by the yolo mode. 

To run this project:
1. donwload it locally 
2. ensure you have docker desktop or docker configured on your sytem 
3. cd into the directory 
4. run docker compose up --build 
5. after the containers are built navigate to http://localhost:8000/docs and use the fastapi gui to interact with the system (alternatively you could use curl commands)
6. to access the firebase database the key is provided in the github for the scope of the project during assessment this may be removed in the future so email bidaouik@gmail.com or k24033682@kcl.ac.uk if you would like access. 

watch the [demo video](https://youtu.be/Ls3KLqmMFrk)