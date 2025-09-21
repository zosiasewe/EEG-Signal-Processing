# Evolutionary-Fuzzy Feature Extraction in EEG Signal Processing for Taste Stimuli

The study involved recording brain activity from participants while they tasted food under two different conditions.

The main question we wanted to answer was:
**Can we tell from brain signals whether someone is tasting some food with their nose-opened or nose-closed?**

##  Data Set

The data set was recorded from 10 participants in a taste stimulus process. It was divided into a class of people eating with their noses closed and a class of people with their noses open.

**Division of the Data Set**:
- **Nose closed** - 5 participants
- **Nose opened** - 5 participants

The project aimed to binary classify the data and see the performance for the taste stimuli process, and also to see how the novel feature extraction method works.
##  Project Overview:
1. Analysis of the data
2. Preprocessing of the EEG data
3. Novel Feature Extraction method
4. Feature Selection
5. Classification
6. Checking which configuration of parameters was the best

**Feature Extraction method:**

EEG data usually contains a lot of information, but it needs translation.
Feature extraction identifies which parts of the brain signal are most important for our task.
Instead of analyzing thousands of data points, we extract meaningful information.
|
A novel (in this field) feature extraction method combining Evolutionary Algorithm with Fuzzy Logic is proposed.

1. Evolutionary-based methodology "described simply"
- It’s an optimization algorithm.
- It works by creating parents and their children.
- It finds the best solution to the given problem.
  In my case, it is used for extracting features.

From our EEG data, we first create some Raw Features. 
**629 features**
Then, from them, we start the evolutionary algorithm process. 
1. We create an initial population by describing it with chromosomes.
   Each chromosome consists of **7 genes**.
   Each of the genes is a representation of some linear & non-linear transformations & weighted combination of them. After that it is mapped to Fuzzy Membership Degrees.

   Example:
   Original Value (Alpha Power (8-13Hz)):	0.75 μV²
   After Normalization:	-0.920 (z-score)
   |
   Total linear combination:	0.212 (includes all features)
   |
   Exponential Component:	tanh(-0.2×0.212 + -0.2) = -0.238
   Sin Component:	sin(2.1×0.212 + 0.5) = 0.811
   Log Component:	log(1.5×0.212 + 1) = 0.276
   Power Component:	0.212^1.2 = 0.155
   |
   Final Extracted Feature:	0.189
   |
   Next step - Fuzzy Membership Mapping
   (It's just an example for you to understand):
   <img width="639" height="441" alt="image" src="https://github.com/user-attachments/assets/160bd678-9162-4e2e-9e99-70ec87acda5d" />

   |
   Our Alpha Power (8-13Hz) value of 0.75 μV² becomes:
   - Fuzzy feature #1 (LOW): 0.000
   - Fuzzy feature #2 (MEDIUM): 0.658
   - Fuzzy feature #3 (HIGH): 0.000
  
   It gives us a robust representation of our feature. It tells us that our Alpha Power is strongly MEDIUM! It gives us more information about each feature, and will help us to create more meaningful features.





