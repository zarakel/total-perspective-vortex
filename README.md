build cmd :
    docker compose build

train cmd : # subject -> Differents sujets de 1 a X #runs -> différents extraits de sujets numéroté  
    docker compose run --rm matplotlib python tpv.py train --subject 5 --runs 3 5 2 6 --model-path qd_model.joblib --show-raw

predict cmd : # subject -> Differents sujets de 1 a X #runs -> différents extraits de sujets numérotés
    docker compose run --rm matplotlib python tpv.py predict --subject 5 --runs 3 5 2 6  --model-path qd_model.joblib