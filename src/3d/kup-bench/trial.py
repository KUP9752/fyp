#%%
dones = [True, False, True, False, False, False]


s = "string.png"
s += ".png"
s

#%%
import ast
import torch


def parse_tensor_list(s: str) -> list[torch.Tensor | None]:
  s = s.strip()[1:-1]  # Remove the outer square brackets
  items = []
  i = 0
  while i < len(s):
      if s[i:i+6] == 'tensor':
          # Find the full tensor(...) expression
          start = i + 7  # after 'tensor('
          depth = 1
          j = start
          while j < len(s):
              if s[j] == '(':
                  depth += 1
              elif s[j] == ')':
                  depth -= 1
                  if depth == 0:
                      break
              j += 1
          tensor_str = s[start:j]
          values = ast.literal_eval(tensor_str)
          items.append(torch.tensor(values))
          i = j + 1  # move past the closing ')'
      elif s[i:i+4] == 'None':
          items.append(None)
          i += 4
      else:
          i += 1  # Skip commas, spaces, etc.
  return items
s = "[tensor([[0.6318, 0.3682]]), None,  tensor([[0.2887, 0.7113]]), tensor([[0.2464, 0.7536]]), tensor([[0.5636, 0.4364]]), tensor([[0.1483, 0.8517]]), tensor([[0.6969, 0.3031]]), tensor([[0.2080, 0.7920]]), tensor([[0.1431, 0.8569]]), tensor([[0.7101, 0.2899]]), None]"

parsed = parse_tensor_list(s)
for item in parsed:
    print(item) 