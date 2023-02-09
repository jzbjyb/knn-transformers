import requests
import logging

def ai21_generate(prompts, temperature=0, max_tokens=256, stop='\n\n'):
    generations = []
    for prompt in prompts:
        try:
            response = requests.post(
                                    "https://api.ai21.com/studio/v1/j1-jumbo/complete",
                                    headers={"Authorization": "Bearer 3lRJs0miElc0d9ZqDzNi8sYrmFD3Ip9f"},
                                    json={
                                            "prompt": prompt, 
                                            "numResults": 1, 
                                            "maxTokens": max_tokens, 
                                            "stopSequences": [stop],
                                            "topKReturn": 0,
                                            "temperature": temperature
                                        }
                                    )


            resp = response.json()
            generation_text = resp['completions'][0]['data']['text']
            finish_reason = resp['completions'][0]['finishReason']['reason']
        except:
            print(response)
            generation_text = 'ERROR'
            finish_reason = 'stop'
            logging.info('Error')

        generations.append((generation_text, finish_reason))

    return generations


if __name__ == "__main__":
    print(ai21_generate(['''Question: Do hamsters provide food for any animals?
Hamsters are prey animals.
Prey are food for predators.
So the final answer is: yes.

Question: Could Brooke Shields succeed at University of Pennsylvania?
Brooke Shields went to Princeton University.
Princeton is ranked as the number 1 national college by US news.
University of Pennsylvania is ranked as number 6 national college by US news.
Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania.
So the final answer is: yes.

Question: Hydrogen's atomic number squared exceeds number of Spice Girls?
Hydrogen has an atomic number of 1.
There are 5 Spice Girls.
1 squared is 1. Thus, Hydrogen's atomic number squared is less than 5.
So the final answer is: no.

Question: Is it common to see frost during some college commencements?
Frost usually can be seen in the winter.
College commencement ceremonies can happen in December, May, and June.
December is in the winter, so there can be frost. Thus, there could be frost at some commencements.
So the final answer is: yes.

Question: Could a llama birth twice during War in Vietnam (1945-46)?
The War in Vietnam was 6 months.
The gestation period for a llama is 11 months.
2 times 11 months is 22 months. 6 months is not longer than 22 months.
So the final answer is: no.

Question: Would a pear sink in water?
The density of a pear is about 0.59 g/cm^3.
The density of water is about 1 g/cm^3.
0.59 g/cm^3 is not greater than 1 g/cm^3? Thus, a pear would float.
So the final answer is: no.

Question: Could a dichromat probably easily distinguish chlorine gas from neon gas?''']))