import cohere

co = cohere.Client('y5mqYHjjsXehMX97OdW66U0KRrGcdrHEM3IHq0Cz') # This is your trial API key
response = co.generate(
  model='command-xlarge-nightly',
  prompt='Question: Do hamsters provide food for any animals?\nHamsters are prey animals.\nPrey are food for predators.\nSo the final answer is: yes.\n\nQuestion: Could Brooke Shields succeed at University of Pennsylvania?\nBrooke Shields went to Princeton University.\nPrinceton is ranked as the number 1 national college by US news.\nUniversity of Pennsylvania is ranked as number 6 national college by US news.\nPrinceton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania.\nSo the final answer is: yes.\n\nQuestion: Hydrogen\'s atomic number squared exceeds number of Spice Girls?\nHydrogen has an atomic number of 1.\nThere are 5 Spice Girls.\n1 squared is 1. Thus, Hydrogen\'s atomic number squared is less than 5.\nSo the final answer is: no.\n\nQuestion: Is it common to see frost during some college commencements?\nFrost usually can be seen in the winter.\nCollege commencement ceremonies can happen in December, May, and June.\nDecember is in the winter, so there can be frost. Thus, there could be frost at some commencements.\nSo the final answer is: yes.\n\nQuestion: Could a llama birth twice during War in Vietnam (1945-46)?\nThe War in Vietnam was 6 months.\nThe gestation period for a llama is 11 months.\n2 times 11 months is 22 months. 6 months is not longer than 22 months.\nSo the final answer is: no.\n\nQuestion: Would a pear sink in water?\nThe density of a pear is about 0.59 g/cm^3.\nThe density of water is about 1 g/cm^3.\n0.59 g/cm^3 is not greater than 1 g/cm^3? Thus, a pear would float.\nSo the final answer is: no.\n\nQuestion: Could a dichromat probably easily distinguish chlorine gas from neon gas?',
  max_tokens=300,
  temperature=0,
  k=0,
  p=1,
  num_generations=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop_sequences=['\n\n'],
  return_likelihoods='GENERATION')

print(response)
exit()
print('Prediction: {}'.format(response.generations[0].text))
