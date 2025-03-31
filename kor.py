from pyswip import Prolog

# Prolog 엔진 생성
prolog = Prolog()

# Prolog 파일 consult
prolog.consult("facts.pl")

# 자손(john, X) 질의 - 한글 predicate 사용
results = list(prolog.query("직책(X)"))

print("존의 자손:")
for result in results:
    print(result["X"])
