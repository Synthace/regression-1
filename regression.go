package regression

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

var (
	errNotEnoughData = errors.New("Not enough data points")
	errTooManyvars   = errors.New("Not enough observations to to support this many variables")
	errRegressionRun = errors.New("Regression has already been run")
)

type Regression struct {
	names             describe
	Data              []*dataPoint
	coeff             map[int]float64
	R2                float64
	Varianceobserved  float64
	VariancePredicted float64
	initialised       bool
	Formula           string
	crosses           []featureCross
	hasRun            bool
}

type dataPoint struct {
	Observed  float64
	Variables []float64
	Predicted float64
	Error     float64
}

type describe struct {
	obs  string
	vars map[int]string
}

// DataPoints is a slice of *dataPoint .
// This type allows for easier construction of training Data points
type DataPoints []*dataPoint

// Creates a new dataPoint
func DataPoint(obs float64, vars []float64) *dataPoint {
	return &dataPoint{Observed: obs, Variables: vars}
}

// Predict updates the "Predicted" value for the input dataPoint
func (r *Regression) Predict(vars []float64) (float64, error) {
	if !r.initialised {
		return 0, errNotEnoughData
	}

	// apply any features crosses to vars
	for _, cross := range r.crosses {
		vars = append(vars, cross.Calculate(vars)...)
	}

	p := r.Coeff(0)
	for j := 1; j < len(r.Data[0].Variables)+1; j++ {
		p += r.Coeff(j) * vars[j-1]
	}
	return p, nil
}

// Set the name of the observed value
func (r *Regression) SetObserved(name string) {
	r.names.obs = name
}

// GetObserved gets the name of the observed value
func (r *Regression) GetObserved() string {
	return r.names.obs
}

// Set the name of variable i
func (r *Regression) SetVar(i int, name string) {
	if len(r.names.vars) == 0 {
		r.names.vars = make(map[int]string, 5)
	}
	r.names.vars[i] = name
}

// GetVar gets the name of variable i
func (r *Regression) GetVar(i int) string {
	x := r.names.vars[i]
	if x == "" {
		s := []string{"X", strconv.Itoa(i)}
		return strings.Join(s, "")
	}
	return x
}

// Registers a feature cross to be applied to the Data points.
func (r *Regression) AddCross(cross featureCross) {
	r.crosses = append(r.crosses, cross)
}

// Train the regression with some Data points
func (r *Regression) Train(d ...*dataPoint) {
	r.Data = append(r.Data, d...)
	if len(r.Data) > 2 {
		r.initialised = true
	}
}

// Apply any feature crosses, generating new observations and updating the Data points, as well as
// populating variable names for the feature crosses.
// this should only be run once, as part of Run().
func (r *Regression) applyCrosses() {
	unusedVariableIndexCursor := len(r.Data[0].Variables)
	for _, point := range r.Data {
		for _, cross := range r.crosses {
			point.Variables = append(point.Variables, cross.Calculate(point.Variables)...)
		}
	}

	if len(r.names.vars) == 0 {
		r.names.vars = make(map[int]string, 5)
	}
	for _, cross := range r.crosses {
		unusedVariableIndexCursor += cross.ExtendNames(r.names.vars, unusedVariableIndexCursor)
	}
}

// Run the regression
func (r *Regression) Run() error {
	if !r.initialised {
		return errNotEnoughData
	}
	if r.hasRun {
		return errRegressionRun
	}

	//apply any features crosses
	r.applyCrosses()
	r.hasRun = true

	observations := len(r.Data)
	numOfvars := len(r.Data[0].Variables)

	if observations < (numOfvars + 1) {
		return errTooManyvars
	}

	// Create some blank variable space
	observed := mat.NewDense(observations, 1, nil)
	variables := mat.NewDense(observations, numOfvars+1, nil)

	for i := 0; i < observations; i++ {
		observed.Set(i, 0, r.Data[i].Observed)
		for j := 0; j < numOfvars+1; j++ {
			if j == 0 {
				variables.Set(i, 0, 1)
			} else {
				variables.Set(i, j, r.Data[i].Variables[j-1])
			}
		}
	}

	// Now run the regression
	_, n := variables.Dims() // cols
	qr := new(mat.QR)
	qr.Factorize(variables)
	q := qr.QTo(nil)
	reg := qr.RTo(nil)

	qtr := q.T()
	qty := new(mat.Dense)
	qty.Mul(qtr, observed)

	c := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		c[i] = qty.At(i, 0)
		for j := i + 1; j < n; j++ {
			c[i] -= c[j] * reg.At(i, j)
		}
		c[i] /= reg.At(i, i)
	}

	// Output the regression results
	r.coeff = make(map[int]float64, numOfvars)
	for i, val := range c {
		r.coeff[i] = val
		if i == 0 {
			r.Formula = fmt.Sprintf("Predicted = %.2f", val)
		} else {
			r.Formula += fmt.Sprintf(" + %v*%.2f", r.GetVar(i-1), val)
		}
	}

	r.calcPredicted()
	r.calcVariance()
	r.calcR2()
	return nil
}

// Coeff returns the calculated coefficient for variable i
func (r *Regression) Coeff(i int) float64 {
	if len(r.coeff) == 0 {
		return 0
	}
	return r.coeff[i]
}

func (r *Regression) calcPredicted() string {
	observations := len(r.Data)
	var predicted float64
	var output string
	for i := 0; i < observations; i++ {
		r.Data[i].Predicted, _ = r.Predict(r.Data[i].Variables)
		r.Data[i].Error = r.Data[i].Predicted - r.Data[i].Observed

		output += fmt.Sprintf("%v. observed = %v, Predicted = %v, Error = %v", i, r.Data[i].Observed, predicted, r.Data[i].Error)
	}
	return output
}

func (r *Regression) calcVariance() string {
	observations := len(r.Data)
	var obtotal, prtotal, obvar, prvar float64
	for i := 0; i < observations; i++ {
		obtotal += r.Data[i].Observed
		prtotal += r.Data[i].Predicted
	}
	obaverage := obtotal / float64(observations)
	praverage := prtotal / float64(observations)

	for i := 0; i < observations; i++ {
		obvar += math.Pow(r.Data[i].Observed-obaverage, 2)
		prvar += math.Pow(r.Data[i].Predicted-praverage, 2)
	}
	r.Varianceobserved = obvar / float64(observations)
	r.VariancePredicted = prvar / float64(observations)
	return fmt.Sprintf("N = %v\nVariance observed = %v\nVariance Predicted = %v\n", observations, r.Varianceobserved, r.VariancePredicted)
}

func (r *Regression) calcR2() string {
	r.R2 = r.VariancePredicted / r.Varianceobserved
	return fmt.Sprintf("R2 = %.2f", r.R2)
}

func (r *Regression) calcResiduals() string {
	str := fmt.Sprintf("Residuals:\nobserved|\tPredicted|\tResidual\n")
	for _, d := range r.Data {
		str += fmt.Sprintf("%.2f|\t%.2f|\t%.2f\n", d.Observed, d.Predicted, d.Observed-d.Predicted)
	}
	str += "\n"
	return str
}

// Display a dataPoint as a string
func (d *dataPoint) String() string {
	str := fmt.Sprintf("%.2f", d.Observed)
	for _, v := range d.Variables {
		str += fmt.Sprintf("|\t%.2f", v)
	}
	return str
}

// Display a regression as a string
func (r *Regression) String() string {
	if !r.initialised {
		return errNotEnoughData.Error()
	}
	str := fmt.Sprintf("%v", r.GetObserved())
	for i := 0; i < len(r.names.vars); i++ {
		str += fmt.Sprintf("|\t%v", r.GetVar(i))
	}
	str += "\n"
	for _, d := range r.Data {
		str += fmt.Sprintf("%v\n", d)
	}
	fmt.Println(r.calcResiduals())
	str += fmt.Sprintf("\nN = %v\nVariance observed = %v\nVariance Predicted = %v", len(r.Data), r.Varianceobserved, r.VariancePredicted)
	str += fmt.Sprintf("\nR2 = %v\n", r.R2)
	return str
}

// MakeDataPoints makes a `[]*dataPoint` from a `[][]float64`. The expected fomat for the input is a row-major [][]float64.
// That is to say the first slice represents a row, and the second represents the cols.
// Furthermore it is expected that all the col slices are of the same length.
// The obsIndex parameter indicates which column should be used
func MakeDataPoints(a [][]float64, obsIndex int) []*dataPoint {
	if obsIndex != 0 && obsIndex != len(a[0])-1 {
		return perverseMakeDataPoints(a, obsIndex)
	}

	retVal := make([]*dataPoint, 0, len(a))
	if obsIndex == 0 {
		for _, r := range a {
			retVal = append(retVal, DataPoint(r[0], r[1:]))
		}
		return retVal
	}

	// otherwise the observation is expected to be the last col
	last := len(a[0]) - 1
	for _, r := range a {
		retVal = append(retVal, DataPoint(r[last], r[:last]))
	}
	return retVal
}

func perverseMakeDataPoints(a [][]float64, obsIndex int) []*dataPoint {
	retVal := make([]*dataPoint, 0, len(a))
	for _, r := range a {
		obs := r[obsIndex]
		others := make([]float64, 0, len(r)-1)
		for i, c := range r {
			if i == obsIndex {
				continue
			}
			others = append(others, c)
		}
		retVal = append(retVal, DataPoint(obs, others))
	}
	return retVal
}
