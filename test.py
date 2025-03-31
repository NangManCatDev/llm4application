from pyswip import Prolog

# Prolog 엔진 생성
prolog = Prolog()

# Prolog 파일 consult
prolog.consult("facts.pl")

# 질의 실행
results = list(prolog.query("ancestor(john, X)"))

# 결과 출력
print("john의 자손:")
for result in results:
    print(result["X"])
