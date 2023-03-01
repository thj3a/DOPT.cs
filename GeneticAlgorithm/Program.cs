// See https://aka.ms/new-console-template for more information
using MatFileHandler;
using System.Diagnostics;
using System.Drawing;
using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;


var rng = new Random();
var path = $"./../../../experiment.json";
var json = File.ReadAllText(path);
Parameters parameters = JsonSerializer.Deserialize<Parameters>(json);

IMatFile matFile;
using (var fileStream = new System.IO.FileStream("./../../../instances/Instance_300_1.mat", System.IO.FileMode.Open))
{
  var reader = new MatFileReader(fileStream);
  matFile = reader.Read();
}
//torch.InitializeDeviceType(DeviceType.OPENCL);
//var device = torch.device("opencl");

var device = torch.device("cpu");
var instance = matFile.Variables.ToDictionary(x => x.Name);
int[] d = matFile["A"].Value.Dimensions;

double[] rawA = matFile["A"].Value.ConvertToDoubleArray();
double[,] A = new double[d[0], d[1]];

for (int i = 0; i < d[0]; i++)
{
  for (int j = 0; j < d[1]; j++)
  {
    A[i, j] = rawA[(i + j * d[0])];
  }
}
double[] x = new double[d[0]];
for (int i = 0; i < x.Length; i++)
{
  if (i < d[0] / 2)
    x[i] = 1;
}

// Tensor tensorA = tensor(matFile["A"].Value.ConvertToDoubleArray(), device: device, dtype: ScalarType.Float64).reshape(new long[] { d[1], d[0] }).T;
Tensor tensorA = tensor(A);
Tensor tensorR = tensor(matFile["R"].Value.ConvertToDoubleArray(), dtype: torch.int64).flatten();
Tensor tensorX = tensor(x);

var times = new List<long>();
var runs = 1_000;
var stopwatch = new Stopwatch();

// var results = torch.linalg.slogdet(mm(mm(tensorA.T, torch.diag(x)), tensorA));
var AT = Transpose(A);
var diagX = Diagonal(x);


static double[,] Transpose(double[,] Mat)
{
  int dim1 = Mat.GetLength(0);
  int dim2 = Mat.GetLength(1);
  double[,] RMat = new double[dim2, dim1];
  for (int i = 0; i < dim2; i++)
  {
    for (int j =0; j < dim1; j++)
    {
      RMat[i, j] = Mat[j, i];
    }
  }
  return RMat;
}
static double[,] Diagonal(double[] x)
{
  int dim = x.Length;
  double[,] RMat = new double[dim, dim];
  for (int i = 0; i < dim; i++)
  {
    for (int j = 0; j < dim; j++)
    {
      if (i == j)
        RMat[i, j] = x[i];
      else
        RMat[i, j] = 0;
    }
  }
  return RMat;
}
static double DET(double[,] Mat)
{
  int n = Mat.GetLength(0);
  if (n != Mat.GetLength(1))
    throw new Exception("Dimensions of matrix must be equal");
  
  double d = 0;
  int k, i, j, subi, subj;
  double[,] SUBMat = new double[n, n];

  if (n == 2)
  {
    return ((Mat[0, 0] * Mat[1, 1]) - (Mat[1, 0] * Mat[0, 1]));
  }

  else
  {
    for (k = 0; k < n; k++)
    {
      subi = 0;
      for (i = 1; i < n; i++)
      {
        subj = 0;
        for (j = 0; j < n; j++)
        {
          if (j == k)
          {
            continue;
          }
          SUBMat[subi, subj] = Mat[i, j];
          subj++;
        }
        subi++;
      }
      d = d + (Math.Pow(-1, k) * Mat[0, k] * DET(SUBMat));
    }
  }
  return d;
}
static double determinant(double[,] array)
{
  double det = 0;
  double total = 0;
  double[,] tempArr = new double[array.GetLength(0) - 1, array.GetLength(1) - 1];

  if (array.GetLength(0) == 2)
  {
    det = array[0, 0] * array[1, 1] - array[0, 1] * array[1, 0];
  }

  else
  {

    for (int i = 0; i < 1; i++)
    {
      for (int j = 0; j < array.GetLength(1); j++)
      {
        tempArr = fillNewArr(array, 0, j);
        det = determinant(tempArr) * (Math.Pow(-1, j) * array[0, j]);
        total += det; ;
      }
    }
  }
  return det;
}
static double[,] fillNewArr(double[,] originalArr, int row, int col)
{
  double[,] tempArray = new double[originalArr.GetLength(0) - 1, originalArr.GetLength(1) - 1];

  for (int i = 0, newRow = 0; i < originalArr.GetLength(0); i++)
  {
    if (i == row)
      continue;
    for (int j = 0, newCol = 0; j < originalArr.GetLength(1); j++)
    {
      if (j == col) continue;
      tempArray[newRow, newCol] = originalArr[i, j];

      newCol++;
    }
    newRow++;
  }
  return tempArray;

}
static double[,] MultiplyMatrix(double[,] A, double[,] B)
{
  int rA = A.GetLength(0);
  int cA = A.GetLength(1);
  int rB = B.GetLength(0);
  int cB = B.GetLength(1);

  if (cA != rB)
    throw new Exception("Number of columns of matrix A must equal number of rows of matrix B");
  
  double temp = 0;
  double[,] kHasil = new double[rA, cB];

  for (int i = 0; i < rA; i++)
  {
    for (int k = 0; k < cA; k++)
    {
      for (int j = 0; j < cB; j++)
      {
        kHasil[i, j] += A[i, k] * B[k, j];
      }
    }
  }
  //Parallel.For(0, rA, i =>
  //{
  //  Parallel.For(0, cA, k =>
  //  {
  //    Parallel.For(0, cB, j =>
  //    {
  //      kHasil[i, j] += A[i, k] * B[k, j];
  //    });
  //  });
  // });

  return kHasil;
}

int maxGenerations = 10_000;
int maxTime = 1800 * 1000;
int populationSize = 100;
double offspringProportion = .5;
double eliteProportion = .3;
double mutationRate = .1;
double crossoverRate = .9;
double best_objval = double.NegativeInfinity;

int offspringSize = (int)Math.Ceiling(offspringProportion * populationSize);
int eliteSize = (int)Math.Ceiling(eliteProportion * populationSize);

List<Chromosome> offspring = new(offspringSize+eliteSize);
List<Chromosome> elite = new(eliteSize);
List<Chromosome> common = new(populationSize-eliteSize);

var sw = new Stopwatch();
var sw_crossover = new Stopwatch();
var sw_selection = new Stopwatch();
var sw_mutation = new Stopwatch();
var sw_evaluation = new Stopwatch();

List<Chromosome> population = InitializePopulation(populationSize, tensorA, tensorR);


sw_evaluation.Start();
Parallel.ForEach(population, x => x.CalculateFitness());
sw_evaluation.Stop();
Chromosome best_chromosome = population.OrderByDescending(x => x.Fitness).First();
LocalSearch(population.OrderByDescending(x => x.Fitness).Take(1).ToList());

sw.Start();

for (int generation = 0; generation < maxGenerations; generation++)
{
  //if (generation % 5000 == 0)
  //{
  //  var result = LocalSearch(population);
  //  population.AddRange(result);
  //  //population.Add(new Chromosome(population[0].Length, A));
  //}

  while (offspring.Count <= offspringSize)
  {
    if (rng.NextDouble() < crossoverRate)
    {
      sw_selection.Start();
      List<Chromosome> selected = Selection(population, 2, rng, Chromosome.SelectionType.random);
      sw_selection.Stop();
      Chromosome parent1 = selected[0];
      Chromosome parent2 = selected[1];

      sw_crossover.Start();
      List<Chromosome> offsprings = Crossover(parent1, parent2, tensorA);
      sw_crossover.Stop();
      offspring.AddRange(offsprings);
    }

    if (rng.NextDouble() < mutationRate)
    {
      if (offspring.Count > 0)
      {
        sw_mutation.Start();
        Chromosome mutant = Mutate(Selection(offspring, 1, rng, Chromosome.SelectionType.random)[0], tensorA, rng);
        sw_mutation.Stop();
        offspring.Add(mutant);
      }
    }
  }
  
  sw_evaluation.Start();
  Parallel.ForEach(offspring, chromosome => chromosome.CalculateFitness());
  sw_evaluation.Stop();
  population.AddRange(offspring);
  Chromosome best_c = population.OrderByDescending(x => x.Fitness).First();

  if (best_c.Fitness > best_objval)
  {
    best_objval = best_c.Fitness;
    best_chromosome = best_c;
  }

  var avg = population.Select(x => x.Fitness).Average();
  var var = population.Select(x => Math.Pow(x.Fitness - avg, 2)).Sum()/population.Count;
  //Console.WriteLine($"Best value at iteration {generation}: value: {best_c.Fitness}, best ever: {best_objval}, max genes: {population.Max(x=> x.Genes.sum().item<double>())} avg: {avg} var: {var}");

  elite.Clear();
  common.Clear();
  
  sw_selection.Start();
  elite = population.OrderByDescending(x => x.Fitness).Take(eliteSize).ToList();
  common = Selection(population, populationSize-eliteSize, rng, Chromosome.SelectionType.roulette);
  sw_selection.Stop();
  
  population.Clear();
  population.AddRange(elite);
  population.AddRange(common);

  offspring.Clear();
}
sw.Stop();

Console.WriteLine($"Best chromosome sum: {best_chromosome.Genes.sum().item<double>()} value: {best_chromosome.Fitness}, total time (ms): {sw.ElapsedMilliseconds}");
Console.WriteLine($"Crossover time: {sw_crossover.ElapsedMilliseconds}");
Console.WriteLine($"Selection time: {sw_selection.ElapsedMilliseconds}");
Console.WriteLine($"Mutation time: {sw_mutation.ElapsedMilliseconds}");
Console.WriteLine($"Evaluation time: {sw_evaluation.ElapsedMilliseconds}");

static double CalculateFitnessTensor(Tensor A, Tensor x)
{
  var (sign, result) = torch.linalg.slogdet(mm(mm(A.T, torch.diag(x)), A));
  double r;
  if (sign.item<double>() < 0)
  {
    r = double.NegativeInfinity;
  }
  else
  {
    r = result.item<double>();
  }
  return r;
}
static List<Chromosome> LocalSearch(List<Chromosome> population)
{
  List<Chromosome> new_population = new(); 

  foreach (var chromosome in population)
  {
    var sw = new Stopwatch();
    sw.Start();
    double[] x = new double[chromosome.Genes.shape[0]];
    for(int i=0; i<chromosome.Genes.shape[0]; i++)
    {
      x[i] = chromosome.Genes[i].item<double>();
    }
    double fit_x = chromosome.Fitness;
    for (int i=0; i<x.Length; i++)
    {
      for (int j=i+1; j < x.Length; j++)
      {
        if (x[i] != x[j])
        {
          x[i] = 1 - x[i];
          x[j] = 1 - x[j];
          var new_fit = chromosome.CalculateFitness(chromosome.A, tensor(x));
          var soma = x.Sum();
          if (new_fit > fit_x)
          {
            fit_x = new_fit;
            continue;
          }
          else
          {
            x[i] = 1 - x[i];
            x[j] = 1 - x[j];
          }
        }
      }
    }
    new_population.Add(new Chromosome(tensor(x), chromosome.A));
    sw.Stop();
    Console.WriteLine($"Local Search Time: {sw.Elapsed}");
  }
  return new_population;
}
static List<Chromosome> InitializePopulation(int size, Tensor A, Tensor R)
{
  List<Chromosome> population = new List<Chromosome>();

  while (population.Count < size)
  {
    // using R
    population.Add(new Chromosome(R, A, torch.linalg.svd));
    // random 
    population.Add(new Chromosome(A));
  }
  Parallel.ForEach(population, chromosome => chromosome.CalculateFitness());
  return population;
}
static List<Chromosome> Selection(List<Chromosome> population, int size, Random rng, Chromosome.SelectionType sel_type)
{
  List<Chromosome> selected = new(size);
  while (selected.Count < size)
  {
    switch (sel_type)
    {
      case Chromosome.SelectionType.roulette: 
        var p_roulette = population.OrderBy(_ => rng.NextDouble()).RandomElementByWeight(x => Math.Abs(1 / x.Fitness));
        if (p_roulette != null)
        {
          selected.Add(p_roulette);
        }
        break;
      case Chromosome.SelectionType.random:
        var p_random = population.OrderBy(x => rng.NextDouble()).Take(size);
        selected.AddRange(p_random);
        break;
      case Chromosome.SelectionType.n_best:
        var p_nbest = population.OrderBy(x => x.Fitness).Take(size);
        selected.AddRange(p_nbest);
        break;
      default:
        break;
    }

  }
  return selected;
}
static List<Chromosome> Crossover(Chromosome parent1, Chromosome parent2, Tensor A)
{
  var range = arange(parent1.Length);
  double s = parent1.Length / 2;
  int n = parent1.Length;
  long cut = randint(parent1.Length, 1).item<long>();

  Tensor genes1 = cat(new List<Tensor>() { parent1.Genes[arange(cut)], parent2.Genes[arange(cut, n)] });
  Tensor genes2 = cat(new List<Tensor>() { parent2.Genes[arange(cut)], parent1.Genes[arange(cut, n)] });

  if (genes1.sum().item<double>() > s)
  {
    int size = (int)(genes1.sum().item<double>() - s);
    Tensor ones = arange(n)[genes1 == 1];
    ones = shuffle_select(ones, size);
    genes1[ones] = 1 - genes1[ones];
  }
  else if (genes1.sum().item<double>() < s)
  {
    int size = (int)(s - genes1.sum().item<double>());
    Tensor zeros = arange(n)[genes1 == 0];
    zeros = shuffle_select(zeros, size);
    genes1[zeros] = 1 - genes1[zeros];
  }
  if (genes2.sum().item<double>() > s)
  {
    int size = (int)(genes2.sum().item<double>() - s);
    Tensor ones = arange(n)[genes2 == 1];
    ones = shuffle_select(ones, size);
    genes2[ones] = 1 - genes2[ones];
  }
  else if (genes2.sum().item<double>() < s)
  {
    int size = (int)(s - genes2.sum().item<double>());
    Tensor zeros = arange(n)[genes2 == 0];
    zeros = shuffle_select(zeros, size);
    genes2[zeros] = 1 - genes2[zeros];
  }

  var c1 = new Chromosome(genes1, A);
  var c2 = new Chromosome(genes2, A);

  return new List<Chromosome>() { c1, c2 };

  //if (new Random().NextDouble() > rate)
  //{
  //  return new Chromosome(parent1.Genes);
  //}

  //int crossoverPoint = new Random().Next(0, parent1.Length - 1);
  //int[] offspringGenes = new int[parent1.Length];

  //for (int i = 0; i < crossoverPoint; i++)
  //{
  //  offspringGenes[i] = parent1.Genes[i];
  //}
  //for (int i = crossoverPoint; i < parent2.Length; i++)
  //{
  //  offspringGenes[i] = parent2.Genes[i];
  //}

  //return new Chromosome(offspringGenes);
}
static Chromosome Mutate(Chromosome chromosome, Tensor A, Random rng)
{
  Tensor genes = chromosome.Genes.clone();
  Tensor ones = arange(genes.shape[0])[genes == 1];
  Tensor zeros = arange(genes.shape[0])[genes == 0];
  ones = shuffle_select(ones, 1);
  zeros = shuffle_select(zeros, 1);

  genes[ones] = 1 - genes[ones];
  genes[zeros] = 1 - genes[zeros];

  var clone = new Chromosome(genes, A);

  return clone;
}
static Tensor shuffle_select(Tensor tensor, int n)
{
  Tensor clone = tensor.clone();
  var perm = randperm(clone.shape[0]);
  clone = clone[perm];
  return clone[arange(n)];
}
class Chromosome
{
  public int Length { get; private set; }
  public Tensor Genes { get; private set; }
  public Tensor A { get; private set; }
  public double Fitness { get; private set; }

  public enum SelectionType
  {
    random = 1,
    roulette = 2,
    n_best = 3
  }

  public Chromosome(Tensor matrix_a)
  {
    this.A = matrix_a;
    int length = (int)this.A.shape[0];
    var x = torch.cat(new List<Tensor>() { torch.ones(length / 2, dtype: ScalarType.Float64), torch.zeros(length / 2, dtype: ScalarType.Float64) });
    Tensor X = x[torch.randperm(length)].clone();
    Genes = X;
    Length = (int)Genes.shape[0];
    //CalculateFitness();
  }
  public Chromosome(Tensor R, Tensor matrix_A, Func<Tensor, bool, (Tensor U, Tensor S, Tensor Vh)> svd)
  {
    this.A = matrix_A;
    var (U, S, Vh) = svd(matrix_A, true);
    long s = A.shape[0] / 2;
    long m = A.shape[0];
    long n = A.shape[1];
    Tensor xbar = torch.sum(torch.pow(U, 2), 1);
    
    Tensor x = torch.zeros(A.shape[0], dtype: ScalarType.Float64);
    x[R] = (double)1;
    xbar[R] = (double)0;
    var phi = torch.argsort(xbar, 0, descending: true).slice(0, 0, (long)xbar.sum().item<double>(), 1)[torch.randperm((long)xbar.sum().item<double>())][arange(s - n)];
    x[phi] = (double)1;

    this.Genes = x;
    this.Length = (int)x.shape[0];
    //CalculateFitness();
  }
  public Chromosome(Tensor genes, Tensor matrix_a)
  {
    A = matrix_a;
    Genes = genes;
    Length = (int)Genes.shape[0];
    //CalculateFitness();
  }

  public void CalculateFitness()
  {
    var (sign, result) = torch.linalg.slogdet(mm(mm(this.A.T, torch.diag(this.Genes)), A));
    double r;
    if (sign.item<double>() < 0)
    {
      r =  double.NegativeInfinity;
    }
    else
    {
      r = result.item<double>();
    }
    Fitness = r;
  }

  public double CalculateFitness(Tensor A, Tensor x)
  {
    var (sign, result) = torch.linalg.slogdet(mm(mm(A.T, torch.diag(x)), A));
    if (sign.item<double>() < 0)
    {
      return double.NegativeInfinity;
    }
    else
    {
      return result.item<double>();
    }
  }


  public override string ToString()
  {
    return string.Join("", Genes);
  }

  public double this[int index]
  {
    get { return Genes[index].item<double>(); }
    set { Genes[index] = value; }
  }

}
public static class IEnumerableExtensions
{
  public static T RandomElementByWeight<T>(this IEnumerable<T> sequence, Func<T, double> weightSelector)
  {
    double totalWeight = sequence.Sum(weightSelector);
    // The weight we are after...
    double itemWeightIndex = new Random().NextDouble() * totalWeight;
    double currentWeightIndex = 0;

    foreach (var item in from weightedItem in sequence select new { Value = weightedItem, Weight = weightSelector(weightedItem) })
    {
      currentWeightIndex += item.Weight;

      // If we've hit or passed the weight we are after for this item then it's the one we want....
      if (currentWeightIndex >= itemWeightIndex)
        return item.Value;

    }

    return default(T);

  }

}
class Parameters
{
  public int max_generations { get; set; }
  public int max_time { get; set; }
  public int population_size { get; set; }
  public float crossover_probability { get; set; }
  public float mutation_probability { get; set; }
  public float elite_size { get; set; }
  public float offspring_size { get; set; }
}

// Mutex mut = new();

// (bool, string) validate_experiment_params(Dictionary<string, string> parameters)
// {
//   List<string> initialization_methods = new List<string>() { "random", "biased", "biasedweighted", "heuristics" };
//   List<string> selection_methods = new List<string>() { "roulette", "tournament", "ranking", "byclass", "fullyrandom", "nbest", "nbestdifferent" };
//   List<string> binary_crossover_methods = new List<string>() { "singlepoint", "mask" };
//   List<string> binary_mutation_methods = new List<string>() { "singlepointlagrangian", "singlepoint", "percentchange", "variablepercentchange" };
//   List<string> permutation_crossover_methods = new List<string>() { "opx", "lox" };
//   List<string> permutation_mutation_methods = new List<string>() { "singleexchange", "percentchange", "variablepercentchange" };
//   return (true, "");
// }

