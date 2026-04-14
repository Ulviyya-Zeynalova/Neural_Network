#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <cstdlib>
#include <ctime>
using namespace std;
class point {
	public:
	double x,y;
	int label;
	};
class norm {
	public:
	double xmin,xmax,ymin,ymax;
	};
double sigmoida(double x) {
    return 1.0/(1.0+exp(-x));
	}
double sigmoida2(double x) {
    double s=sigmoida(x);
    return s*(1.0-s);
	}
vector<point> read(const string &file) { 
    vector<point> vec;
    ifstream f(file);
    if (!f) {
        cout<<"failed to open file"<<endl;
        return vec;
		}
    point p;
    while (f>>p.x>>p.y>>p.label) vec.push_back(p);
    return vec;
	}
norm normp(const vector<point> &traindata) {
    norm obb;
    if (traindata.empty()) {
        obb={0,0,0,0};
        return obb;
		}
    obb.xmin=obb.xmax=traindata[0].x;
    obb.ymin=obb.ymax=traindata[0].y;
    for(const auto& p : traindata) {
        if (p.x<obb.xmin) obb.xmin=p.x;
        if (p.x>obb.xmax) obb.xmax=p.x;
        if (p.y<obb.ymin) obb.ymin=p.y;
        if (p.y>obb.ymax) obb.ymax=p.y;
		}
    return obb;
	}
void normirovka(double &x, double &y, const norm &obb) {
    double dx=obb.xmax-obb.xmin;
    double dy=obb.ymax-obb.ymin;
    if (dx==0.0) dx=1.0;
    if (dy==0.0) dy=1.0;
    x=2.0*(x-obb.xmin)/dx-1.0;
    y=2.0*(y-obb.ymin)/dy-1.0;
	}	
void split(vector<point> &alldata, vector<point> &traindata, vector<point> &validdata) { 
    random_device rd;
    mt19937 g(rd());
    shuffle(alldata.begin(), alldata.end(), g);
    int trainsize=(int)(alldata.size()*0.8);
    traindata.assign(alldata.begin(), alldata.begin()+trainsize);
    validdata.assign(alldata.begin()+trainsize, alldata.end());
	}
class neuralnetwork {
    norm params;
    vector<int> layers; 
    vector<vector<vector<double>>> w; 
    vector<vector<double>> b;        
    vector<vector<double>> a;     
    vector<vector<double>> f; 
    vector<vector<double>> delta; 
    double lr;  
	public:
	void set_norm(const norm &p) {params=p;} 
    neuralnetwork(const vector<int> &inner_size, double learning_rate) {
        lr=learning_rate;
        layers.push_back(2); 
        for(auto &h : inner_size)        
        layers.push_back(h);
        layers.push_back(1);
        weights();
		}
    void weights() {
		int l=layers.size();
        w.resize(l);
        b.resize(l);
        for(int k=1;k<l;k++) { 
            int prev=layers[k-1]; 
            int curr=layers[k];
            b[k].resize(curr);
            for (int i=0;i<curr; i++) b[k][i] = 0.0;
			w[k].resize(curr);
			for (int i=0;i<curr;i++) {
				w[k][i].resize(prev);
				for (int j=0;j<prev;j++) {
					w[k][i][j]=rand()/(double)RAND_MAX*2-1;
					}
				}
			}
		}
	double forward(double x, double y) {
		normirovka(x, y, params);
		int l = layers.size();
		a.resize(l);
		f.resize(l);
		a[0].resize(2);
		a[0][0] = x;
		a[0][1] = y;
		for(int k = 1; k < l; k++) {
			int prevn = layers[k-1];
			int currn = layers[k];
			f[k].resize(currn);
			a[k].resize(currn);
			for(int i = 0; i < currn; i++) {
				f[k][i] = b[k][i];
				for(int j = 0; j < prevn; j++) f[k][i] += w[k][i][j] * a[k-1][j];
				a[k][i] = sigmoida(f[k][i]);
				}
			}
		return a[l-1][0];
		}
	void tofile(const vector<point> &data,ofstream &file) {
		for(const auto &p : data) {
			point temp = p;              
			int pred=predict(temp);     
			file<<p.x<<" "<<p.y<<" "<<pred<<endl;
			}
		}
	void backward (int truelabel) {
		int l=layers.size();
		delta.resize(l);
		delta[l-1].resize(1); 
		double p=a[l-1][0];
		double dL_dp;
		if(truelabel==1) dL_dp=-1.0/p;
		else dL_dp = 1.0/(1.0-p);
		delta[l-1][0]=dL_dp*p*(1.0-p);
		for(int k=l-2;k>=1;k--) {
			int currn=layers[k];
			int nextn=layers[k+1];
			delta[k].resize(currn);
			for(int i=0;i<currn;i++) {
				delta[k][i]=0.0;
				for(int j=0;j<nextn;j++) delta[k][i]+=w[k+1][j][i]*delta[k+1][j];
				delta[k][i]*=sigmoida2(f[k][i]);
				}
			}
		}
	void train_epoch(vector<point> &traindata) {
		random_device rd;
		mt19937 g(rd());
		shuffle(traindata.begin(), traindata.end(), g);
		for(auto &p:traindata) {
			forward(p.x, p.y);
			backward(p.label);
			update();
			}
		}
	void update() {
		int l=layers.size();
		for(int k=1;k<l;k++) {
			int prevn=layers[k-1];
			int currn=layers[k];
			for(int i=0;i<currn;i++) { 
				for(int j=0;j<prevn;j++) w[k][i][j]-=lr*delta[k][i]*a[k-1][j]; 
				b[k][i]-=lr*delta[k][i]; 
				}
			}
		}
	int predict(point &p){ 
		 double prob=forward(p.x,p.y);
		 return (prob>=0.5) ? 1 : 0;
		 }
	double evaluate(const vector<point> &data) {
		int count=0; 
		for(const auto &p:data) {
			point temp=p;
			int pred=predict(temp);
			if(pred==p.label) count++;
			}
		return (double)count/data.size();
		}			
	};

void start(ofstream &f) { 
	    f<<"set xlabel \"x\""<<endl
		<<"set ylabel \"y\""<<endl
		<<"set size ratio -1"<<endl
		<<"set palette defined(0 \"blue\", 1 \"red\")"<<endl
        <<"set key off"<<endl
        <<"plot \"pointnew.txt\" using 1:2:3 with points pt 7 ps 0.1 lc palette"<<endl;
        }


void gen(int n0, int n1, double R, double a) {
    ofstream f("point.txt");
    mt19937 gen(random_device{}());
    uniform_real_distribution<> dist(-a,a);
    int count=0;
    while (count<n0) {
        double x=dist(gen);
        double y=dist(gen);
        if (x*x+y*y<=R*R) {
            f<<x<<" "<<y<<" "<<0<<endl;
            count++;
			}
		}
    count=0;
    while (count<n1) {
        double x=dist(gen);
        double y=dist(gen);
        if (x*x+y*y>R*R) {
            f<<x<<" "<<y<<" "<<1<<endl;
            count++;
			}
		}
	}

	
int main() {
	int size;
	cout<<"type the number of dots:"<<endl;
	cin>>size;
	double R=0.5,a=1;
	gen(size/2,size/2,R,a);
	vector<point> alldata=read("point.txt");
    if (alldata.empty()) {
        cout<<"file is empty"<<endl;
        return 1;
		}
	ofstream file("pointnew.txt");
	ofstream f("point.plt");
	start(f);  
    vector<point> traindata, validdata;
    split(alldata, traindata, validdata);
    vector<int> inner = {8,8}; 
    double lr=0.05; 
    neuralnetwork net(inner, lr);
	norm params=normp(traindata);
	net.set_norm(params);
	int epochs=15; 
	for (int e=0;e<epochs;e++) {
	net.train_epoch(traindata);
	double valid_acc=net.evaluate(validdata);
	cout<<e+1<<endl;
	cout<<"valid accuracy: "<<valid_acc<<endl;
	}
	net.tofile(alldata, file);
	return 0;
}
