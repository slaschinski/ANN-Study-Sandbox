# ANN Study Sandbox
A little Unity based sandbox with an ANN (Artificial Neural Network) implemented in C# for experimentation purposes

## Goal of this project
Implement a neural network in a way it's easy to read as a software developer, without any regards to the performance.

## Future of this project
### Your part
If you find any mistakes on the implementation, please let me know. I will also happily accept any contributions in every way, as long as it's toward the goal of this project.
Also if you have some questions trying to understand the code, please write an issue and I'll try to help.
### My part
I still put some effort into this project regarding bug-fixing and documentation. Also there is still some polishing needed.
I might implement one or two more deep learning concepts, but only if it helps building understanding of neural nets. Otherwise existing frameworks will do a much better job with better performance and less bugs than I could ever achieve.

## Status of this project
### Scenes
At the moment there are three scenes present in the project.
Digital Gates - Simple implementations of AND and XOR in a neural net, to test the viability of the code. Most of it needs to be activated first, by activating the script in the corresponding game objects.
BallBalance2D_Q - Uses Q learning to balance a ball on a plane, rotating on one axis. In it's current state the code tries to read an XML file, rather than learning. The path to that XML file need to be specified in ANN.cs at the very bottom. Saving and loading weights is very much work in progress
BallBalance3D_Q - Also uses Q learning to balance a ball on a plane rotating in two axis. While this seems simple, it might be too much for the concept as it is implemented right now. Other concepts like OPP instead of Q-learning might be needed. This one is still up for experimentation

### Code
The ANN code should pretty much be universal as long as some form of gradient descent is used. Other implementations might be possible out of the box or require minor tweaking of the code. The Q learning code on the other hand is pretty specific for its task and it's hard to do a general implementation because it needs to be somewhat specific to the task it's doing. Also I wanted to keep the complexity low.

## Background
This project emerged from my personal studies on neural networks. After making my first steps with Python and TensorFlow, I wanted to gain a deeper understanding of the concepts. I also wanted to do some experimentations using Q-Learning in Unity.
I know that the Unity team is working on it's own implementation of TensorFlow utilizing Python for learning. While this is much more practical and high-performance, it is also much more abstract and gets complex if you take a look under the hood. This is not an attempt to reinvent the wheel. This is an attempt to implement a neural network in a way it's easier to understand as a software developer, who isn't also a great mathematician. Since neural networks gained popularity only some time ago, there isn't a ton of references out there. So I decided to upload my attempts on the topic to maybe help other developers grasping the concept by having more references.
