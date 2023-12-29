``````
input -> (16, 50, 1200)

step 1 : after siren => (16, 50, 64, 132)
step 2 : after 2Dconv => (16, 128, 52, 100)
step 3 : after aggregation (either of them) => 
              method 1 : (16, 128, 5200) :-C (memory issues possibly)
              method 2 : (16, 128, 52, 100) -> (16, 52, 128, 100) -> pointwise-conv [k=1] : (16, 1, 128, 100) -> (16, 128, 100)
              method 3 : (16, 128, 52, 100) -> (16, 52, 128, 100) -> pointwise-conv [k=(d<period)] : (16, d, 128, 100) -> (16, 128, d*100) [best]

output -> (16, 128, d*100)

Onto next Block ...

``````