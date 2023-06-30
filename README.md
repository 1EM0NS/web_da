# API Documentation

## Classify

### POST /classify

Classify the iris dataset based on height and leaf samples.

#### Request Body

```json
{
  "height": 5,
  "leaf_samples": 3
}
```

#### Response Body

```json
{
  "decision_tree": "base64-encoded-image",
  "result_img": "base64-encoded-image",
  "result": [
    [5.1, 3.5, 1.4, 0.2, "setosa"],
    [4.9, 3.0, 1.4, 0.2, "setosa"],
    ...
  ]
}
```

### GET /classify

Get the iris dataset.

#### Response Body

```json
{
  "total": 150,
  "df": [
    [5.1, 3.5, 1.4, 0.2, "setosa"],
    [4.9, 3.0, 1.4, 0.2, "setosa"],
    ...
  ]
}
```

## Cluster

### POST /cluster

Cluster the generated data based on the specified model.

#### Request Body

```json
{
  "model": "kmeans",
  "k": 3
}
```

#### Response Body

```json
{
  "circle": [
    {
      "data": [0.0, 1.0],
      "label": 0
    },
    {
      "data": [0.0, -1.0],
      "label": 1
    },
    ...
  ],
  "blob": [
    {
      "data": [0.0, 1.0],
      "label": 0
    },
    {
      "data": [0.0, -1.0],
      "label": 1
    },
    ...
  ],
  "moon": [
    {
      "data": [0.0, 1.0],
      "label": 0
    },
    {
      "data": [0.0, -1.0],
      "label": 1
    },
    ...
  ],
  "iris": [
    {
      "data": [0.0, 1.0],
      "label": 0
    },
    {
      "data": [0.0, -1.0],
      "label": 1
    },
    ...
  ]
}
```

### GET /cluster

Get the generated data.

#### Response Body

```json
{
  "circle_data": [
    [0.0, 1.0],
    [0.0, -1.0],
    ...
  ],
  "blob_data": [
    [0.0, 1.0],
    [0.0, -1.0],
    ...
  ],
  "moon_data": [
    [0.0, 1.0],
    [0.0, -1.0],
    ...
  ],
  "iris_data": [
    [5.1, 3.5, 1.4, 0.2, "Iris-setosa"],
    [4.9, 3.0, 1.4, 0.2, "Iris-setosa"],
    ...
  ]
}
```

## Apriori

### POST /aprior

Perform association analysis on the dataset.

#### Request Body

```json
{
  "min_support": 0.3,
  "min_confidence": 0.3
}
```

#### Response Body

```json
{
  "label": [
    ["beer"],
    ["bread"],
    ...
  ],
  "confidence": [
    [0.5],
    [0.5],
    ...
  ],
  "support": [
    [0.5],
    [0.5],
    ...
  ]
}
```

### GET /aprior

Get the dataset.

#### Response Body

```json
{
  "data": [
    ["beer", "bread", "diapers"],
    ["beer", "bread", "milk"],
    ...
  ],
  "data_table": [
    {
      "item0": "beer",
      "item1": "bread",
      "item2": "diapers"
    },
    {
      "item0": "beer",
      "item1": "bread",
      "item2": "milk"
    },
    ...
  ]
}
```

## Regression

### POST /regression

Perform regression analysis on the generated data.

#### Request Body

```json
{
  "n_samples": 100,
  "degree": 2
}
```

#### Response Body

```json
{
  "data": [
    [0.0, 0.0],
    [0.010101010101010102, 0.010101010101010102],
    ...
  ],
  "w": [
    0.0,
    1.0,
    2.0
  ],
  "b": 0.0
}
```