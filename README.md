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
<br />
**Feature Extraction method:** <br />
<br />
EEG data usually contains a lot of information, but it needs translation.<br />
Feature extraction identifies which parts of the brain signal are most important for our task.<br />
Instead of analyzing thousands of data points, we extract meaningful information.<br />
<br />
A novel (in this field) feature extraction method combining Evolutionary Algorithm with Fuzzy Logic is proposed.<br />
<br />
1. Evolutionary-based methodology "described simply"<br />
- It’s an optimization algorithm.<br />
- It works by creating parents and their children.<br />
- It finds the best solution to the given problem.<br />
  In my case, it is used for extracting features.<br />
<br />
From our EEG data, we first create some Raw Features. <br />
**629 features**<br />
Then, from them, we start the evolutionary algorithm process. <br />
1. We create an initial population by describing it with chromosomes.<br />
   Each chromosome consists of **7 genes**.<br />
   Each of the genes is a representation of some linear & non-linear transformations & weighted combination of them. After that, it is mapped to Fuzzy Membership Degrees.<br />
<br />
   Example:<br />
   Original Value (Alpha Power (8-13Hz)):	0.75 μV²<br />
   After Normalization:	-0.920 (z-score)<br />
   <br />
   Total linear combination:	0.212 (includes all features)<br />
   <br />
   Exponential Component:	tanh(-0.2×0.212 + -0.2) = -0.238<br />
   Sin Component:	sin(2.1×0.212 + 0.5) = 0.811<br />
   Log Component:	log(1.5×0.212 + 1) = 0.276<br />
   Power Component:	0.212^1.2 = 0.155<br />
   <br />
   Final Extracted Feature:	0.189<br />
   <br />
   Next step - Fuzzy Membership Mapping<br />
   (It's just an example for you to understand): <br />
   <img width="639" height="441" alt="image" src="https://github.com/user-attachments/assets/160bd678-9162-4e2e-9e99-70ec87acda5d" />
<br />
   <br />
   Our Alpha Power (8-13Hz) value of 0.75 μV² becomes:<br />
   - Fuzzy feature #1 (LOW): 0.000<br />
   - Fuzzy feature #2 (MEDIUM): 0.658<br />
   - Fuzzy feature #3 (HIGH): 0.000<br />
  <br />
   It gives us a robust representation of our feature. It tells us that our Alpha Power is strongly MEDIUM! It gives us more information about each feature and will help us to create more meaningful features.





