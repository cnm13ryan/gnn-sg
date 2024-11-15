## ClassDef SemiGroup
```json
{
  "name": "Target",
  "description": "The Target class represents a specific entity within a game environment. It extends the Entity class and includes properties and methods tailored to its functionality.",
  "extends": "Entity",
  "properties": [
    {
      "name": "health",
      "type": "number",
      "description": "Indicates the current health points of the target."
    },
    {
      "name": "maxHealth",
      "type": "number",
      "description": "Represents the maximum health points the target can have."
    },
    {
      "name": "position",
      "type": "Vector3",
      "description": "Stores the current position of the target in 3D space."
    }
  ],
  "methods": [
    {
      "name": "takeDamage",
      "parameters": [
        {
          "name": "amount",
          "type": "number"
        }
      ],
      "returnType": "void",
      "description": "Reduces the target's health by a specified amount. If the health drops to zero or below, the target is marked as defeated."
    },
    {
      "name": "heal",
      "parameters": [
        {
          "name": "amount",
          "type": "number"
        }
      ],
      "returnType": "void",
      "description": "Increases the target's health by a specified amount, ensuring it does not exceed its maximum health."
    },
    {
      "name": "moveToPosition",
      "parameters": [
        {
          "name": "newPosition",
          "type": "Vector3"
        }
      ],
      "returnType": "void",
      "description": "Updates the target's position to a new specified location in 3D space."
    }
  ]
}
```
