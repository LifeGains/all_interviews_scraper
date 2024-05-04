# Interview Classificaiton Model

- Again solving a problem that I have.
- I want to put all the full interviews/lectures/speeches of successful famous people in one place with a nice timeline for each interview (eg. Steve Jobs, Jeff Bezos, Charlie Munger, Buffett, Sam Altman)
- Working on a classifier that outputs predictions if the person in question is a 1st person participant in the interview, or a 3rd person speaker about the person.
- Tried to use images but 1) the thumbnails on youtube are unreliable
- Using Youtube Descriptions to predict if the person is indeed a 1st person speaker. Used custom data that I labeled myself as training data.
- Slowly building up from baseline model to most advanced:
    - Word2Vec: 49% Macro F1 Score
    - GloVe: 