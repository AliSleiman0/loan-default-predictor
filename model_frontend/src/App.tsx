import React, { useState, type JSX } from "react";
import axios from "axios";
import {
  Container,
  Row,
  Col,
  Form,
  Button,
  Card,
  Spinner,
  ProgressBar,
  Alert,
} from "react-bootstrap";
import "bootstrap/dist/css/bootstrap.min.css";

// Feature shape expected by backend
type Features = {
  Loan_ID: string;
  Gender: string;
  Married: string;
  Dependents?: string; // optional, keep as string ("0","1","2","3+")
  Education: string;
  Self_Employed: string;
  ApplicantIncome: number;
  CoapplicantIncome: number;
  LoanAmount?: number; // optional
  Loan_Amount_Term?: number; // optional
  Credit_History?: number; // optional (1 or 0)
  Property_Area: string;
};

type PredictionResponse = {
  prediction: number; // 0 or 1
  probability: number; // 0..1
};

const initialFeatures: Features = {
  Loan_ID: "LP001015",
  Gender: "Male",
  Married: "Yes",
  Dependents: "0",
  Education: "Graduate",
  Self_Employed: "No",
  ApplicantIncome: 5720,
  CoapplicantIncome: 0,
  LoanAmount: 110,
  Loan_Amount_Term: 360,
  Credit_History: 1,
  Property_Area: "Urban",
};

// numeric keys for typed handler
type NumericKey =
  | "ApplicantIncome"
  | "CoapplicantIncome"
  | "LoanAmount"
  | "Loan_Amount_Term"
  | "Credit_History";

export default function App(): JSX.Element {
  const [features, setFeatures] = useState<Features>(initialFeatures);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Handler for text/select inputs
  function handleChange<K extends keyof Features>(key: K, value: Features[K]) {
    setFeatures((prev) => ({ ...prev, [key]: value }));
  }

  // Handler for numeric inputs (keeps numbers or undefined)
  function handleNumberChange(key: NumericKey, value: string) {
    if (value === "") {
      // set to 0 if empty (you can choose undefined if preferred)
      setFeatures((prev) => ({ ...prev, [key]: 0 } as Features));
      return;
    }
    const parsed = Number(value);
    setFeatures((prev) => ({ ...prev, [key]: isNaN(parsed) ? 0 : parsed } as Features));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);

    // Prepare payload: keep Dependents as string (backend handles '3+' etc)
    const payload = { ...features };

    try {
      const res = await axios.post<PredictionResponse>(
        "http://0.0.0.0:10000/predict",
        payload,
        { headers: { "Content-Type": "application/json" } }
      );

      if (
        res?.data &&
        typeof res.data.prediction === "number" &&
        typeof res.data.probability === "number"
      ) {
        setResult(res.data);
      } else {
        setError("Unexpected response format from server.");
      }
    } catch (err: unknown) {
      console.error(err);
      if (axios.isAxiosError(err) && err.response?.data) {
        setError(`Server error: ${JSON.stringify(err.response.data)}`);
      } else {
        setError(
          "Network or server unreachable. Make sure server is running."
        );
      }
    } finally {
      setLoading(false);
    }
  }

  function resetForm() {
    setFeatures(initialFeatures);
    setResult(null);
    setError(null);
  }

  function humanPredictionLabel(pred: number) {
    return pred === 1 ? "Approved" : "Rejected";
  }

  const percentage =
    result && result.probability != null ? Math.round(result.probability * 100) : 0;

  return (
    <Container className="py-4 d-flex justify-content-center">
   

      <Row className="w-100 justify-content-center">
        <Col xs={12} md={10} lg={12}>
          <h3>Loan Prediction</h3>
          <p className="text-muted">
            Fill the form below and click <strong>Predict</strong>. Values will be processed the
            same way as the model training pipeline.
          </p>
        </Col>
        
         
        <Col xs={12} md={10} lg={7}>
          <Card className="mb-3 shadow-sm">
            <Card.Body>
              <Form onSubmit={handleSubmit}>
                <Row>
                  <Col md={6}>
                    <Form.Group className="mb-3">
                      <Form.Label>Gender</Form.Label>
                      <Form.Select
                        value={features.Gender}
                        onChange={(e) => handleChange("Gender", e.target.value)}
                      >
                        <option>Male</option>
                        <option>Female</option>
                      </Form.Select>
                      <Form.Text className="text-muted">Select applicant gender.</Form.Text>
                    </Form.Group>
                  </Col>

                  <Col md={6}>
                    <Form.Group className="mb-3">
                      <Form.Label>Married</Form.Label>
                      <Form.Select
                        value={features.Married}
                        onChange={(e) => handleChange("Married", e.target.value)}
                      >
                        <option>Yes</option>
                        <option>No</option>
                      </Form.Select>
                      <Form.Text className="text-muted">Whether the applicant is married.</Form.Text>
                    </Form.Group>
                  </Col>
                </Row>

                <Row>
                  <Col md={4}>
                    <Form.Group className="mb-3">
                      <Form.Label>Dependents</Form.Label>
                      <Form.Select
                        value={features.Dependents}
                        onChange={(e) => handleChange("Dependents", e.target.value)}
                      >
                        <option value="0">0</option>
                        <option value="1">1</option>
                        <option value="2">2</option>
                        <option value="3+">3+</option>
                      </Form.Select>
                      <Form.Text className="text-muted">Number of dependents </Form.Text>
                    </Form.Group>
                  </Col>

                  <Col md={4}>
                    <Form.Group className="mb-3">
                      <Form.Label>Education</Form.Label>
                      <Form.Select
                        value={features.Education}
                        onChange={(e) => handleChange("Education", e.target.value)}
                      >
                        <option>Graduate</option>
                        <option>Not Graduate</option>
                      </Form.Select>
                      <Form.Text className="text-muted">Applicant education level.</Form.Text>
                    </Form.Group>
                  </Col>

                  <Col md={4}>
                    <Form.Group className="mb-3">
                      <Form.Label>Self Employed</Form.Label>
                      <Form.Select
                        value={features.Self_Employed}
                        onChange={(e) => handleChange("Self_Employed", e.target.value)}
                      >
                        <option>No</option>
                        <option>Yes</option>
                      </Form.Select>
                      <Form.Text className="text-muted">Is the applicant self-employed?</Form.Text>
                    </Form.Group>
                  </Col>
                </Row>

                <Row>
                  <Col md={6}>
                    <Form.Group className="mb-3">
                      <Form.Label>Applicant Income</Form.Label>
                      <Form.Control
                        type="number"
                        value={features.ApplicantIncome}
                        onChange={(e) => handleNumberChange("ApplicantIncome", e.target.value)}
                        placeholder="Monthly applicant income"
                      />
                    
                    </Form.Group>
                  </Col>

                  <Col md={6}>
                    <Form.Group className="mb-3">
                      <Form.Label>Coapplicant Income</Form.Label>
                      <Form.Control
                        type="number"
                        value={features.CoapplicantIncome}
                        onChange={(e) => handleNumberChange("CoapplicantIncome", e.target.value)}
                        placeholder="Monthly coapplicant income (0 if none)"
                      />
                   
                    </Form.Group>
                  </Col>
                </Row>

                <Row>
                  <Col md={6}>
                    <Form.Group className="mb-3">
                      <Form.Label>Loan Amount (in 1000s)</Form.Label>
                      <Form.Control
                        type="number"
                        value={features.LoanAmount ?? ""}
                        onChange={(e) => handleNumberChange("LoanAmount", e.target.value)}
                        placeholder="Loan amount (e.g. 110)"
                      />
                     
                    </Form.Group>
                  </Col>

                  <Col md={6}>
                    <Form.Group className="mb-3">
                      <Form.Label>Loan Term (months)</Form.Label>
                      <Form.Control
                        type="number"
                        value={features.Loan_Amount_Term ?? ""}
                        onChange={(e) => handleNumberChange("Loan_Amount_Term", e.target.value)}
                        placeholder="Term in months (e.g. 360)"
                      />
                    
                    </Form.Group>
                  </Col>
                </Row>

                <Row>
                  <Col md={6}>
                    <Form.Group className="mb-3">
                      <Form.Label>Credit History</Form.Label>
                      <Form.Select
                        value={String(features.Credit_History ?? "")}
                        onChange={(e) => {
                          const v = e.target.value;
                          handleNumberChange("Credit_History", v === "" ? "" : v);
                        }}
                      >
                        <option value="">Unknown</option>
                        <option value="1">1 (meets credit history)</option>
                        <option value="0">0 (no history)</option>
                      </Form.Select>
                      <Form.Text className="text-muted">1 = has credit history, 0 = none.</Form.Text>
                    </Form.Group>
                  </Col>

                  <Col md={6}>
                    <Form.Group className="mb-3">
                      <Form.Label>Property Area</Form.Label>
                      <Form.Select
                        value={features.Property_Area}
                        onChange={(e) => handleChange("Property_Area", e.target.value)}
                      >
                        <option>Urban</option>
                        <option>Semiurban</option>
                        <option>Rural</option>
                      </Form.Select>
                      <Form.Text className="text-muted">Where the property is located.</Form.Text>
                    </Form.Group>
                  </Col>
                </Row>

                <div className="d-flex gap-2 mt-3">
                  <Button variant="primary" type="submit" disabled={loading}>
                    {loading ? (
                      <>
                        <Spinner animation="border" size="sm" /> Predicting...
                      </>
                    ) : (
                      "Predict"
                    )}
                  </Button>
                  <Button variant="outline-secondary" onClick={resetForm} disabled={loading}>
                    Reset
                  </Button>
                </div>
              </Form>
            </Card.Body>
          </Card>

          
        </Col>
        <Col xs={12} md={10} lg={5}>
        {error && (
            <Alert variant="danger" onClose={() => setError(null)} dismissible>
              <strong>Error:</strong> {error}
            </Alert>
          )}

         {result && (
  <Card className="shadow-sm">
    <Card.Body>
      <Row>
        <Col md={8}>
          <h5>Prediction</h5>
          <div className="fs-4">
            {humanPredictionLabel(result.prediction)}{" "}
            <small className="text-muted">({result.prediction})</small>
          </div>
          <div>
            Probability: <strong>{Math.round(result.probability * 100)}%</strong>
          </div>
          <ProgressBar className="mt-2" now={percentage} label={`${percentage}%`} />

          {/* Note for the user */}
          <small className="text-muted d-block mt-2">
            Note: A probability above 50% means the loan is likely to be approved.
            Below 50% means it is likely to be rejected.
          </small>
        </Col>
      </Row>
    </Card.Body>
  </Card>
)}

          </Col>
      </Row>
    </Container>
  );
}
