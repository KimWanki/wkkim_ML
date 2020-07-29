# 1)문자열 분리하기   str.split([sep])
#   split() : sep을 기준으로 문자열을 분리해 리스트로 반환
coffee_menu = "에스프레소,아메리카노,카페라테,카푸치노"
print(coffee_menu.split(','))
print("에스프레소,아메리카노,카페라테,카푸치노".split(','))
print("에스프레소 아메리카노 카페라테 카푸치노".split(' '))

#   sep 입력 없는 경우, 공백 및 개행문자('\n') 제거
print("     에스프레소   \n\n   아메리카노   \n\n   카페라테   \n\n   카푸치노".split())

# 2)원하는 횟수만큼 문자열 분리하기 str.split([sep],maxplit)
#   split(maxplit=2) : maxplit만큼 분리해 리스트로 반환
#   maxsplit 생략 가능.
print("에스프레소 아메리카노 카페라테 카푸치노".split(' ', 1))
print("에스프레소 아메리카노 카페라테 카푸치노".split(' ', maxsplit=2))

