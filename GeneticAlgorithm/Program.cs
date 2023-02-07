// See https://aka.ms/new-console-template for more information
using MatFileHandler;
using System.Diagnostics;
using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

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

var instance = matFile.Variables.ToDictionary(x => x.Name);
int[] d = matFile["A"].Value.Dimensions;


Tensor A = torch.tensor(matFile["A"].Value.ConvertToDoubleArray(), device: CPU, dtype: ScalarType.Float64).reshape(new long[] { d[1], d[0] }).T;
Tensor R = tensor(matFile["R"].Value.ConvertToDoubleArray(), dtype: torch.int64).flatten();

int maxGenerations = 100_000;
int maxTime = 1800 * 1000;
int populationSize = 100;
double offspringProportion = 1;
double eliteProportion = .3;
double mutationRate = .1;
double crossoverRate = .9;
double best_objval = double.NegativeInfinity;
Chromosome best_chromosome;

List<Chromosome> population = InitializePopulation(populationSize, d[0], A, R);
best_chromosome = population[0];

int offspringSize = (int)offspringProportion * populationSize;
int eliteSizeInt = (int)eliteProportion * populationSize;

List<Chromosome> offspring = new(offspringSize+eliteSizeInt);
List<Chromosome> elite = new(eliteSizeInt);
List<Chromosome> common = new(populationSize-eliteSizeInt);

var sw = new Stopwatch();
sw.Start();

for (int generation = 0; generation < maxGenerations; generation++)
{

  if (generation % 1000 == 0)
  {
    for (int i=0;i<populationSize; i++)
    {
      population.Add(new Chromosome(population[0].Length, A));
    }
  }

  while (offspring.Count <= offspringProportion)
  {
    if (rng.NextDouble() < crossoverRate)
    {
      List<Chromosome> selected = Selection(population, 2, rng);
      Chromosome parent1 = selected[0];
      Chromosome parent2 = selected[1];

      List<Chromosome> offsprings = Crossover(parent1, parent2, A);
      offspring.AddRange(offsprings);
    }

    if (rng.NextDouble() < mutationRate)
    {
      if (offspring.Count > 0)
      {
        Chromosome mutant = Mutate(Selection(offspring, 1, rng)[0], A, rng);
        offspring.Add(mutant);
      }
    }
  }



  population.AddRange(offspring);
  Chromosome best_c = population.OrderByDescending(x => x.Fitness).First();

  if (best_c.Fitness > best_objval)
  {
    best_objval = best_c.Fitness;
    best_chromosome = best_c;
  }

  Console.WriteLine($"Best value at iteration {generation}: value: {best_c.Fitness}, best ever: {best_objval}, max genes: {population.Max(x=> x.Genes.sum().item<double>())} mean: {population.Select(x => x.Fitness).Average()}");

  elite = PickElite(population, eliteSizeInt);
  common = Selection(population, populationSize-eliteSizeInt, rng);

  population.Clear();
  population.AddRange(elite);
  population.AddRange(common);
  elite.Clear();
  common.Clear();
  offspring.Clear();
}
sw.Stop();

Console.WriteLine($"Best chromosome sum: {best_chromosome.Genes.sum().item<double>()} value: {best_chromosome.CalculateFitness()}, total time: {sw.Elapsed}");
best_chromosome.Genes.print();

static List<Chromosome> InitializePopulation(int size, int length, Tensor A, Tensor R)
{
  List<Chromosome> population = new List<Chromosome>();

  
  var (U, S, Vh) = torch.linalg.svd(A);
  long s = A.shape[0] / 2;
  long m = A.shape[0];
  long n = A.shape[1];
  Tensor xbar = torch.sum(torch.pow(U, 2), 1);


  for (int i = 0; i < size; i++)
  {
    Tensor x = torch.zeros(A.shape[0], dtype: ScalarType.Float64);
    x[R] = (double)1;
    xbar[R] = (double)0;
    var phi = torch.argsort(xbar, 0, descending: true).slice(0, 0, (long)xbar.sum().item<double>(), 1)[torch.randperm((long)xbar.sum().item<double>())][arange(s - n)];
    x[phi] = (double)1;

    population.Add(new Chromosome(x, A));
    //population.Add(new Chromosome(length, A));
  }
  return population;
}

static List<Chromosome> Selection(List<Chromosome> population, int size, Random rng)
{
  List<Chromosome> selected = new(size);
  while (selected.Count < size)
  {
    // weighted
    var p = population.OrderBy(_ => rng.NextDouble()).RandomElementByWeight(x => Math.Abs(1 / x.Fitness));
    if (p != null)
    {
      selected.Add(p);
    }

    //// n best
    //var p = population.OrderBy(x => x.Fitness).Take(size);
    //selected.AddRange(p);

    //// random
    //var p = population.OrderBy(x => rng.NextDouble()).Take(size);
    //selected.AddRange(p);
  }
  return selected;
}

static List<Chromosome> PickElite(List<Chromosome> population, int size)
{
  return population.OrderByDescending(x => x.Fitness).Take(size).ToList();
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

  public Chromosome(int length, Tensor matrix_a)
  {
    A = matrix_a;
    var x = torch.cat(new List<Tensor>() { torch.ones(length / 2, dtype: ScalarType.Float64), torch.zeros(length / 2, dtype: ScalarType.Float64) });
    Tensor X = x[torch.randperm(length)].clone();
    Genes = X;
    Length = (int)Genes.shape[0];
    var f = CalculateFitness();
    Fitness = f;
  }

  public Chromosome(Tensor genes, Tensor matrix_a)
  {
    A = matrix_a;
    Genes = genes;
    Length = (int)Genes.shape[0];
    Fitness = CalculateFitness();
  }

  public double CalculateFitness()
  {
    var sw = new Stopwatch();
    var (sign, result) = torch.linalg.slogdet(mm(mm(A.T, torch.diag(Genes)), A));
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

