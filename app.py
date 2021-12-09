from flask import Flask,render_template,request
import model as m
app=Flask(__name__)
@app.route("/",methods=["GET","POST"])
def first():
    News=""
    Label=""
    t_d=""
    nlabel=""
    dnews=""
    if request.method=="POST":
        News=request.form['news']
        print(News)
        print(type(News))
        t_d=m.test(News)
        nlabel=t_d
        print(nlabel)
    return render_template("lstm.html",t_d=nlabel,dnews=News)
    
if __name__=="__main__":
    app.run(debug=True)
